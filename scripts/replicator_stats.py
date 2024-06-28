from datetime import datetime, timedelta
from pprint import pprint

import typer
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)

from prediction_market_agent.agents.replicate_to_omen_agent.deploy import (
    REPLICATOR_ADDRESS,
)


def main() -> None:
    markets = OmenSubgraphHandler().get_omen_binary_markets(
        limit=None,
        creator=REPLICATOR_ADDRESS,
    )

    stats = {
        "markets created": len(markets),
        "open markets": len([m for m in markets if m.is_open]),
        "liquidity in open markets": sum(
            [
                OmenAgentMarket.from_data_model(m).get_liquidity().amount
                for m in markets
                if m.is_open
            ]
        ),
        "liquidity in closed markets": sum(
            [
                OmenAgentMarket.from_data_model(m).get_liquidity().amount
                for m in markets
                if not m.is_open
            ]
        ),
        "liquidity in open markets closing in more than 30 days": sum(
            [
                OmenAgentMarket.from_data_model(m).get_liquidity().amount
                for m in markets
                if m.opening_datetime > datetime.now() + timedelta(days=30)
            ]
        ),
        "liquidity in open markets closing in more than 180 days": sum(
            [
                OmenAgentMarket.from_data_model(m).get_liquidity().amount
                for m in markets
                if m.opening_datetime > datetime.now() + timedelta(days=180)
            ]
        ),
    }
    pprint(stats)


if __name__ == "__main__":
    typer.run(main)
