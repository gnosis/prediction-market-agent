from datetime import timedelta

import typer
from prediction_market_agent_tooling.tools.omen.reality_accuracy import reality_accuracy

from prediction_market_agent.agents.ofvchallenger_agent.deploy import (
    MARKET_CREATORS_TO_CHALLENGE,
    OFV_CHALLENGER_SAFE_ADDRESS,
)


def main(since_days: int) -> None:
    since = timedelta(days=since_days)
    ofv_challenger_accuracy = reality_accuracy(OFV_CHALLENGER_SAFE_ADDRESS, since)
    olas_accuracies = {
        addr: reality_accuracy(addr, since) for addr in MARKET_CREATORS_TO_CHALLENGE
    }

    print(
        f"OFVChallenger accuracy: {ofv_challenger_accuracy.accuracy*100:.2f}% out of {ofv_challenger_accuracy.total}"
    )
    for idx, accuracy_report in enumerate(olas_accuracies.values()):
        if accuracy_report.total:
            print(
                f"Olas {idx} accuracy: {accuracy_report.accuracy*100:.2f}% out of {accuracy_report.total}"
            )
        else:
            print(f"Olas {idx} accuracy: no answers found")


if __name__ == "__main__":
    typer.run(main)
