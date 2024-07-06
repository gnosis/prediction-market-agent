import sys

import typer
from prediction_market_agent_tooling.deploy.agent import Answer
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.metaculus.metaculus import (
    MetaculusAgentMarket,
)

from prediction_market_agent.agents.think_thoroughly_agent.think_thoroughly_agent import (
    CrewAIAgentSubquestions,
)

WARMUP_TOURNAMENT_ID = 3294

APP = typer.Typer(pretty_exceptions_enable=False)


@APP.command()
def main(
    dummy_prediction: bool = False,
    repeat_predictions: bool = False,
    tournament_id: int = WARMUP_TOURNAMENT_ID,
):
    """
    Submit predictions to Metaculus markets using the Think Thoroughly agent.

    https://www.metaculus.com/notebooks/25525/announcing-the-ai-forecasting-benchmark-series--july-8-120k-in-prizes/
    """

    model: str = "gpt-4-turbo-2024-04-09"
    agent = CrewAIAgentSubquestions(model=model, memory=False)
    markets: list[MetaculusAgentMarket] = MetaculusAgentMarket.get_binary_markets(
        limit=sys.maxsize,
        tournament_id=tournament_id,
        filter_by=FilterBy.OPEN,
        sort_by=SortBy.NEWEST,
    )
    logger.info(f"Found {len(markets)} open markets to submit predictions for.")

    if not repeat_predictions:
        # Filter out markets that we have already answered
        markets = [market for market in markets if not market.have_predicted]
        logger.info(
            f"Found {len(markets)} unanswered markets to submit predictions for."
        )

    for market in markets:
        logger.info(f"Answering market {market.id}, question: {market.question}")
        if dummy_prediction:
            answer: Answer = Answer(
                p_yes=0.5,
                decision=True,
                reasoning="Just a test.",
                confidence=0.5,
            )
        else:
            # TODO incorporate 'Resolution criteria', 'Fine print', and
            # 'Background info' into the prompt given to the agent.
            answer = agent.answer_binary_market(
                market.question, created_time=market.created_time
            )
            if answer is None:
                logger.info("No answer was given. Skipping")

        market.submit_prediction(p_yes=answer.p_yes, reasoning=answer.reasoning)


if __name__ == "__main__":
    APP()
