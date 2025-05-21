import typing as t
from datetime import timedelta

from prediction_market_agent_tooling.deploy.agent import (
    MAX_AVAILABLE_MARKETS,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.deploy.trade_interval import (
    FixedInterval,
    TradeInterval,
)
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from web3 import Web3

from prediction_market_agent.agents.prophet_agent.deploy import (
    DeployablePredictionProphetGPT4oAgent,
    DeployablePredictionProphetGPTo1PreviewAgent,
)

# List of white-listed market creator addresses that these specialized agents will bet on.
SPECIALIZED_FOR_MARKET_CREATORS: list[ChecksumAddress] = [
    Web3.to_checksum_address("0xa7E93F5A0e718bDDC654e525ea668c64Fd572882"),
]


class GetMarketCreatorsStalkerMarkets:
    # Do as many bets as we can for the special markets.
    bet_on_n_markets_per_run = MAX_AVAILABLE_MARKETS
    n_markets_to_fetch: int = MAX_AVAILABLE_MARKETS
    # These tends to be long-running markets, it's not interesting to bet on them too much.
    same_market_trade_interval: TradeInterval = FixedInterval(timedelta(days=14))
    supported_markets: t.Sequence[MarketType] = [MarketType.OMEN]

    def get_markets(
        self,
        market_type: MarketType,
        sort_by: SortBy = SortBy.CLOSING_SOONEST,
        filter_by: FilterBy = FilterBy.OPEN,
    ) -> t.Sequence[OmenAgentMarket]:
        available_markets = [
            OmenAgentMarket.from_data_model(m)
            for m in OmenSubgraphHandler().get_omen_markets_simple(
                limit=self.n_markets_to_fetch,
                sort_by=sort_by,
                filter_by=filter_by,
                creator_in=SPECIALIZED_FOR_MARKET_CREATORS,
            )
        ]

        return available_markets


class MarketCreatorsStalkerAgent1(
    GetMarketCreatorsStalkerMarkets, DeployablePredictionProphetGPTo1PreviewAgent
):
    pass


class MarketCreatorsStalkerAgent2(
    GetMarketCreatorsStalkerMarkets, DeployablePredictionProphetGPT4oAgent
):
    pass
