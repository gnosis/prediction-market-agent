import multiprocessing
import os
import time

import typer
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType

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


def monitor_processes(processes: dict[str, multiprocessing.Process]) -> None:
    while True:
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


def main() -> None:
    agent_table_handler = AgentTableHandler()
    all_agents: list[AgentDB] = list(agent_table_handler.sql_handler.get_all())

    processes: dict[str, multiprocessing.Process] = {}
    for agent in all_agents:
        processes[agent.name] = spawn_process(agent.name)
        processes[agent.name].start()

    try:
        monitor_processes(processes)
    except KeyboardInterrupt:
        logger.info("Stopping all agents...")
        stop_all_processes(processes)


if __name__ == "__main__":
    typer.run(main)
