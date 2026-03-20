from prediction_market_agent_tooling.benchmark.benchmark import Benchmarker
from prediction_market_agent_tooling.benchmark.agents import RandomAgent

from scripts.benchmark_wrappers import (
    MultiPersonaBenchmarkedAgent,
    MultiPersonaEnsembleBenchmarkedAgent,
)

from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy

from prediction_market_agent_tooling.markets.markets import get_binary_markets
from prediction_market_agent_tooling.markets.market_type import MarketType



def dedupe_markets_by_question(markets):
    seen = set()
    deduped = []
    for m in markets:
        if m.question not in seen:
            seen.add(m.question)
            deduped.append(m)
    return deduped


def load_markets(limit=50):
    #handler = OmenSubgraphHandler()

    #markets = handler.get_omen_markets_simple(
    #    limit=limit*3,
    #    filter_by=FilterBy.RESOLVED,
    #    sort_by=SortBy.NEWEST,
    #)

    markets = get_binary_markets(
        limit=limit,
        market_type=MarketType.OMEN,
    )

    # keep only resolved markets
    markets = [m for m in markets if not m.is_resolved()]
    markets = dedupe_markets_by_question(markets)


    return markets[:limit]


def main():
    markets = load_markets(limit=50)

    agents = [
        RandomAgent(agent_name="random"),
        MultiPersonaBenchmarkedAgent(),
        MultiPersonaEnsembleBenchmarkedAgent(),
    ]

    benchmarker = Benchmarker(
        markets=markets,
        agents=agents
        #cache_path="benchmark_cache.json",
    )

    benchmarker.run_agents()

    metrics = benchmarker.compute_metrics()

    print("\n=== BENCHMARK RESULTS ===\n")
    for key, values in metrics.items():
        print(f"{key}: {values}")

    report = benchmarker.generate_markdown_report()

    with open("benchmark_report.md", "w") as f:
        f.write(report)

    print("\nSaved → benchmark_report.md")


if __name__ == "__main__":
    main()
