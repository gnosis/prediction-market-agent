from prediction_market_agent_tooling.markets.agent_market import AgentMarket


def market_is_saturated(market: AgentMarket) -> bool:
    return market.current_p_yes > 0.95 or market.current_p_no > 0.95
