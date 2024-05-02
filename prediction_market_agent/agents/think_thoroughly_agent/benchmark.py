import typing as t
from datetime import datetime, timedelta

import typer
from prediction_market_agent_tooling.benchmark.agents import (
    AbstractBenchmarkedAgent,
    FixedAgent,
    RandomAgent,
)
from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.benchmark.utils import (
    OutcomePrediction,
    Prediction,
)
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.markets.markets import (
    MarketType,
    get_binary_markets,
)
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.agents.think_thoroughly_agent.deploy import (
    DeployableThinkThoroughlyAgent,
)


def build_binary_agent_market_from_question(question: str) -> AgentMarket:
    return AgentMarket(
        id=question,
        url=question,
        close_time=utcnow() + timedelta(days=1),
        volume=None,
        question=question,
        current_p_yes=Probability(0.5),
        created_time=datetime(2024, 1, 1),
        resolution=None,
        outcomes=["YES", "NO"],
    )


class CrewAIAgentSubquestionsBenchmark(AbstractBenchmarkedAgent):
    def __init__(
        self,
        max_workers: int,
        agent_name: str,
    ) -> None:
        self.agent = DeployableThinkThoroughlyAgent().agent
        super().__init__(agent_name=agent_name, max_workers=max_workers)

    def predict(self, market_question: str) -> Prediction:
        result = self.agent.answer_binary_market(market_question)
        return Prediction(
            outcome_prediction=(
                OutcomePrediction(
                    decision=result.decision,
                    p_yes=result.p_yes,
                    confidence=result.confidence,
                    info_utility=None,
                )
                if result
                else None
            )
        )


def main(
    n: int = 50,
    output: str = "./benchmark_report_50markets.md",
    reference: MarketType = MarketType.MANIFOLD,
    filter: FilterBy = FilterBy.OPEN,
    sort: SortBy = SortBy.NONE,
    max_workers: int = 1,
    cache_path: t.Optional[str] = "predictions_cache.json",
    only_cached: bool = False,
) -> None:
    """
    Polymarket usually contains higher quality questions,
    but on Manifold, additionally to filtering by MarketFilter.resolved, you can sort by MarketSort.newest.
    """
    markets = get_binary_markets(n, reference, filter_by=filter, sort_by=sort)
    markets_deduplicated = list(({m.question: m for m in markets}.values()))
    if len(markets) != len(markets_deduplicated):
        logger.debug(
            f"Warning: Deduplicated markets from {len(markets)} to {len(markets_deduplicated)}."
        )

    logger.debug(f"Found {len(markets_deduplicated)} markets.")

    benchmarker = Benchmarker(
        markets=markets_deduplicated,
        agents=[
            CrewAIAgentSubquestionsBenchmark(
                agent_name="subsequential-questions-crewai",
                max_workers=max_workers,
            ),
            RandomAgent(agent_name="random", max_workers=max_workers),
            FixedAgent(
                fixed_answer=False, agent_name="fixed-no", max_workers=max_workers
            ),
            FixedAgent(
                fixed_answer=True, agent_name="fixed-yes", max_workers=max_workers
            ),
        ],
        cache_path=cache_path,
        only_cached=only_cached,
    )

    benchmarker.run_agents(
        enable_timing=False
    )  # Caching of search etc. can distort timings
    md = benchmarker.generate_markdown_report()

    with open(output, "w") as f:
        logger.info(f"Writing benchmark report to: {output}")
        f.write(md)


if __name__ == "__main__":
    typer.run(main)
