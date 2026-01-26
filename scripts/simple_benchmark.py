import typing as t
from datetime import timedelta

import typer
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.benchmark.utils import Prediction
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.gtypes import (
    HexBytes,
    OutcomeStr,
    OutcomeToken,
    Probability,
)
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.data_models import (
    CategoricalProbabilisticAnswer,
    ProbabilisticAnswer,
)
from prediction_market_agent_tooling.markets.markets import (
    FilterBy,
    MarketType,
    SortBy,
    get_binary_markets,
)
from prediction_market_agent_tooling.markets.omen.data_models import Condition
from prediction_market_agent_tooling.markets.omen.omen import (
    MarketFees,
    OmenAgentMarket,
)
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    WrappedxDaiContract,
)
from prediction_market_agent_tooling.tools.utils import utcnow
from web3 import Web3

from prediction_market_agent.agents.advanced_agent.deploy import AdvancedAgent
from prediction_market_agent.agents.coinflip_agent.deploy import DeployableCoinFlipAgent


def main(
    n: int = 10,
    output: str = "./benchmark_report.md",
    cache_path: t.Optional[str] = None,
    only_cached: bool = False,
) -> None:
    markets = get_binary_markets(
        n, MarketType.MANIFOLD, filter_by=FilterBy.OPEN, sort_by=SortBy.NONE
    )
    markets_deduplicated = list(({m.question: m for m in markets}.values()))

    print(f"Found {len(markets_deduplicated)} markets.")

    benchmarker = Benchmarker(
        markets=markets_deduplicated,
        agents=[
            BenchmarkAgent(agent=AdvancedAgent()),
            BenchmarkAgent(agent=DeployableCoinFlipAgent()),
        ],
        cache_path=cache_path,
        only_cached=only_cached,
    )

    benchmarker.run_agents()
    md = benchmarker.generate_markdown_report()

    with open(output, "w") as f:
        print(f"Writing benchmark report to: {output}")
        f.write(md)


class BenchmarkAgent(AbstractBenchmarkedAgent):
    def __init__(self, agent: DeployableTraderAgent) -> None:
        super().__init__(agent_name=agent.__class__.__name__, max_workers=1)
        self.agent = agent

    def predict(self, market_question: str) -> Prediction:
        try:
            output = self.agent.answer_binary_market(
                market=OmenAgentMarket(
                    question=market_question,
                    # Rest of the fields are not used in this benchmark, so we can just fill them with dummy values.
                    id="id",
                    creator="creator",  # type: ignore # dummy input
                    outcomes=[OutcomeStr("Yes"), OutcomeStr("No")],
                    probabilities={
                        OutcomeStr("Yes"): Probability(0.5),
                        OutcomeStr("No"): Probability(0.5),
                    },
                    collateral_token_contract_address_checksummed=WrappedxDaiContract().address,
                    market_maker_contract_address_checksummed=Web3.to_checksum_address(
                        "0xf3318C420e5e30C12786C4001D600e9EE1A7eBb1"
                    ),
                    created_time=utcnow() - timedelta(days=1),
                    close_time=utcnow(),
                    resolution=None,
                    condition=Condition(id=HexBytes("0x123"), outcomeSlotCount=2),
                    url="url",
                    volume=None,
                    finalized_time=None,
                    fees=MarketFees.get_zero_fees(bet_proportion=0.02),
                    outcome_token_pool={
                        OutcomeStr("Yes"): OutcomeToken(1000),
                        OutcomeStr("No"): OutcomeToken(1000),
                    },
                )
            )
        except ValueError as e:
            print(f"Failed to predict for market by {self.agent_name}: {e}")
            output = None
        if output is None:
            print(
                f"Failed to predict for market by {self.agent_name}: {market_question}"
            )
            return Prediction()
        return Prediction(
            is_predictable=True,
            outcome_prediction=CategoricalProbabilisticAnswer.from_probabilistic_answer(
                ProbabilisticAnswer(
                    p_yes=output.p_yes,
                    confidence=output.confidence,
                )
            ),
        )


if __name__ == "__main__":
    typer.run(main)
