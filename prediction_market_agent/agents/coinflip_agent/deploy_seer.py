from datetime import timedelta

from prediction_market_agent_tooling.deploy.trade_interval import (
    TradeInterval,
    FixedInterval,
)
from prediction_market_agent_tooling.markets.agent_market import SortBy

from prediction_market_agent.agents.coinflip_agent.deploy import DeployableCoinFlipAgent


class DeployableSeerCoinFlipAgent(DeployableCoinFlipAgent):
    n_markets_to_fetch = 5
    get_markets_sort_by = SortBy.HIGHEST_LIQUIDITY
    bet_on_n_markets_per_run = 2
    same_market_trade_interval: TradeInterval = FixedInterval(timedelta(days=14))
