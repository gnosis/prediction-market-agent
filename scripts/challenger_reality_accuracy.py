from collections import defaultdict
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.dates import AutoDateLocator, DateFormatter
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.markets.omen.data_models import (
    HexBytes,
    RealityResponse,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenSubgraphHandler
from prediction_market_agent_tooling.tools.omen.reality_accuracy import reality_accuracy
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow

from prediction_market_agent.agents.ofvchallenger_agent.deploy import (
    MARKET_CREATORS_TO_CHALLENGE,
    OFV_CHALLENGER_SAFE_ADDRESS,
)


def main(
    since_days: int,
    challenger: ChecksumAddress = OFV_CHALLENGER_SAFE_ADDRESS,
    market_creators: list[ChecksumAddress] | None = None,
) -> None:
    market_creators = check_not_none(market_creators or MARKET_CREATORS_TO_CHALLENGE)
    since = timedelta(days=since_days)
    ofv_challenger_accuracy = reality_accuracy(challenger, since)
    olas_accuracies = {addr: reality_accuracy(addr, since) for addr in market_creators}

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

    plot_number_of_challenges(challenger, since)


def plot_number_of_challenges(challenger: ChecksumAddress, since: timedelta) -> None:
    all_responses_on_challenged_questions = OmenSubgraphHandler().get_responses(
        limit=None,
        question_id_in=list(
            set(
                r.question.questionId
                for r in OmenSubgraphHandler().get_responses(
                    limit=None,
                    user=challenger,
                    question_finalized_before=utcnow(),
                    question_finalized_after=utcnow() - since,
                )
            )
        ),
    )

    question_id_to_responses: dict[HexBytes, list[RealityResponse]] = defaultdict(list)
    for response in all_responses_on_challenged_questions:
        question_id_to_responses[response.question.questionId].append(response)

    # Prepare data for plotting
    dates = []
    counts = []

    for responses in question_id_to_responses.values():
        first_response = responses[0]
        # Could be None if pending arbitration, let's just ignore those.
        if first_response.question.answer_finalized_datetime is not None:
            dates.append(first_response.question.answer_finalized_datetime)
            counts.append(len(responses))

    # Add jitter to counts and dates to reduce overlap
    counts_jittered = [count + np.random.uniform(-0.1, 0.1) for count in counts]
    dates_jittered = [
        date + timedelta(minutes=np.random.uniform(-120, 120)) for date in dates
    ]  # Jitter up to 2 hours

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(
        dates_jittered, counts_jittered, alpha=0.6, edgecolors="w", linewidth=0.5  # type: ignore[arg-type] # It works with datetime but mypy complains.
    )

    # Set the title and labels
    plt.title("Number of Responses Over Time")
    plt.xlabel("Question Finalization Time")
    plt.ylabel("Number of Responses")

    # Improve date formatting on the x-axis
    date_format = DateFormatter("%Y-%m-%d")
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.gca().xaxis.set_major_locator(AutoDateLocator())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    typer.run(main)
