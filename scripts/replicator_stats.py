from datetime import datetime, timedelta
from pprint import pprint

import typer
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.parallelism import par_generator
from tqdm import tqdm

from prediction_market_agent.agents.replicate_to_omen_agent.deploy import (
    REPLICATOR_ADDRESS,
)


def main() -> None:
    now = datetime.now()
    markets = OmenSubgraphHandler().get_omen_binary_markets(
        limit=None,
        creator=REPLICATOR_ADDRESS,
    )
    bets_for_market = {
        market.id: bets
        for market, bets in tqdm(
            par_generator(
                markets,
                lambda m: (
                    m,
                    OmenSubgraphHandler().get_bets(
                        market_id=m.market_maker_contract_address_checksummed
                    ),
                ),
            ),
            total=len(markets),
            desc="Loading bets",
        )
    }

    open_markets_closing_in_less_than_30_days = [
        m for m in markets if now < m.opening_datetime < now + timedelta(days=30)
    ]
    markets_closing_in_more_than_30_days = [
        m for m in markets if m.opening_datetime > now + timedelta(days=30)
    ]

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
                for m in markets_closing_in_more_than_30_days
            ]
        ),
        "liquidity in open markets closing in more than 180 days": sum(
            [
                OmenAgentMarket.from_data_model(m).get_liquidity().amount
                for m in markets
                if m.opening_datetime > datetime.now() + timedelta(days=180)
            ]
        ),
        "avg bets per market closing in less than 30 days": sum(
            len(bets_for_market[m.id])
            for m in open_markets_closing_in_less_than_30_days
        )
        / len(open_markets_closing_in_less_than_30_days),
        "avg bets per market closing in more than 30 days": sum(
            len(bets_for_market[m.id]) for m in markets_closing_in_more_than_30_days
        )
        / len(markets_closing_in_more_than_30_days),
    }
    pprint(stats)


if __name__ == "__main__":
    typer.run(main)
