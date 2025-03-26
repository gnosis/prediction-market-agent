import typer

from prediction_market_agent.agents.safe_guard_agent.safe_guard import validate_all


def main(
    do_sign_or_execution: bool = typer.Option(
        False, help="Execute transaction if validated"
    ),
    do_reject: bool = typer.Option(False, help="Reject transaction if not validated"),
    do_message: bool = typer.Option(False, help="Send a message about the outcome"),
) -> None:
    validate_all(
        do_sign_or_execution=do_sign_or_execution,
        do_reject=do_reject,
        do_message=do_message,
    )


if __name__ == "__main__":
    typer.run(main)
