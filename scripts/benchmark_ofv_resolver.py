import pandas as pd
import typer
from prediction_market_agent_tooling.tools.caches.inmemory_cache import (
    persistent_inmemory_cache,
)

from prediction_market_agent.agents.ofvchallenger_agent.ofv_resolver import (
    ofv_answer_binary_question,
)
from prediction_market_agent.utils import APIKeys

APP = typer.Typer()


@persistent_inmemory_cache
def ofv_answer_binary_question_cached(question: str) -> bool | None:
    result = ofv_answer_binary_question(question, APIKeys())
    return result.factuality if result is not None else None


@APP.command()
def full(data_path: str) -> None:
    """
    Will run the OFV resolver on all provided data.
    Expects a tsv file with columns:
        - question
        - resolution (YES/NO, as currently resolved on Omen)
        - my_resolution (YES/NO, as resolved manually by you, used as ground truth)
    """
    df = pd.read_csv(data_path, sep="\t")

    # Run the resolution on all the data.
    df["ofv_resolution"] = df["question"].apply(
        lambda q: ofv_answer_binary_question_cached(q)
    )
    # Normalise boolean to YES/NO/None.
    df["ofv_resolution"] = df["ofv_resolution"].apply(
        lambda r: "None" if r is None else "YES" if r else "NO"
    )
    # Save all the predictions and separatelly these that are incorrect.
    df.to_csv("markets_resolved.tsv", sep="\t", index=False)
    df[df["ofv_resolution"] != df["my_resolution"]].to_csv(
        "markets_resolved_incorretly_by_ofv.tsv", sep="\t", index=False
    )

    # Calculate the accuracy.
    accuracy_current = sum(df["resolution"] == df["my_resolution"]) / len(df)
    accuracy_ofv = sum(df["ofv_resolution"] == df["my_resolution"]) / len(df)
    print(
        f"""
Current accuracy: {accuracy_current*100:.2f}%
OFV's accuracy: {accuracy_ofv*100:.2f}%
"""
    )


@APP.command()
def single(question: str) -> None:
    """
    Will run the prediction market resolver and print the result on a single question.
    """
    ofv_answer_binary_question(
        question,
        api_keys=APIKeys(),
    )


if __name__ == "__main__":
    APP()
