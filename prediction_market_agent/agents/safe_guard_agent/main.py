import time

import typer
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.deploy.agent import initialize_langfuse
from prediction_market_agent_tooling.loggers import logger

from prediction_market_agent.agents.safe_guard_agent.safe_guard import validate_all


def main(
    do_sign_or_execution: bool = typer.Option(
        False, help="Execute transaction if validated"
    ),
    do_reject: bool = typer.Option(False, help="Reject transaction if not validated"),
    do_message: bool = typer.Option(False, help="Send a message about the outcome"),
    run_time_limit: int = 55 * 60,  # seconds
    sleep_between_validations: int = 15,  # seconds
) -> None:
    initialize_langfuse(enable_langfuse=APIKeys().default_enable_langfuse)

    start_time = time.time()
    while time.time() - start_time < run_time_limit:
        validate_all(
            do_sign_or_execution=do_sign_or_execution,
            do_reject=do_reject,
            do_message=do_message,
        )
        logger.info(
            f"Waiting for {sleep_between_validations} seconds before next validation..."
        )
        time.sleep(sleep_between_validations)


if __name__ == "__main__":
    typer.run(main)
