from prediction_market_agent_tooling.config import APIKeys as gtools
from prediction_market_agent_tooling.markets.markets import (
    get_binary_markets,
    MarketType,
)

from prediction_market_agent.agents.known_outcome_agent.deploy import (
    DeployableKnownOutcomeAgent,
)
from prediction_market_agent.agents.think_thoroughly_agent.deploy import (
    DeployableThinkThoroughlyAgent,
)
from prediction_market_agent.utils import APIKeys as Keys1

if __name__ == "__main__":
    print("start")
    keys = Keys1()
    keys2 = gtools()
    print(f"{keys.langfuse_host} {keys2.langfuse_host}")

    agent = DeployableThinkThoroughlyAgent()
    # agent = DeployableKnownOutcomeAgent()

    markets = get_binary_markets(5, MarketType.OMEN)
    market = markets[0]
    answer = agent.answer_binary_market(market)
    bet_amount = agent.calculate_bet_amount(answer, market)

    print(f"Would bet {bet_amount.amount} {bet_amount.currency} on {answer}!")

    print("finished")
