import prediction_market_agent_tooling.benchmark.benchmark as bm
from prediction_market_agent_tooling.benchmark.agents import RandomAgent, ProphetGPT4oAgent
from prediction_market_agent_tooling.markets.market_type import MarketType
from prediction_market_agent_tooling.markets.markets import get_binary_markets
from prediction_market_agent_tooling.benchmark.agents import AbstractBenchmarkedAgent
from agents.multi_persona_ensemble_agent.deploy import MultiPersonaEnsembleAgent

def start_benchmark():
    # 1. Grab 3 real markets from Omen (Presagio)
    print("📡 Fetching markets for benchmark...")
    markets = get_binary_markets(limit=3, market_type=MarketType.OMEN)
    
    # 2. Setup the Benchmarker
    # We compare a 'Random' agent against the 'Wisdom of the Crowd' (Market Price)
    benchmarker = bm.Benchmarker(
        markets=markets,
        agents=[RandomAgent(agent_name="baseline_bot")]
    )
    
    # 3. Run and print results
    print("📊 Running simulation...")
    benchmarker.run_agents()
    
    # This creates a nice markdown table of the results
    report = benchmarker.generate_markdown_report()
    print("\n--- BENCHMARK RESULTS ---")
    print(report)

if __name__ == "__main__":
    start_benchmark()
