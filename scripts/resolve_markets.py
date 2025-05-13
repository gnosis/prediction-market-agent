from datetime import datetime, timezone

import typer
from eth_typing import HexAddress
from eth_typing.evm import HexStr
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)

from prediction_market_agent.agents.ofvchallenger_agent.deploy import OFVChallengerAgent
from prediction_market_agent.utils import APIKeys

MARKET_IDS: list[HexAddress] = [
    HexAddress(HexStr("000000000000000000000000000000000000000000")),
]


def main() -> None:
    api_keys = APIKeys()
    agent = OFVChallengerAgent()

    markets = [
        OmenSubgraphHandler().get_omen_market_by_market_id(market_id)
        for market_id in MARKET_IDS
    ]

    for market in markets:
        logger.info(
            f"Challenging market {market.question.id=} - {market.question.title=} - {market.question.outcomes=}"
        )
        logger.info(f"Market creation timestamp: {market.creation_datetime}")
        logger.info(f"Market close time: {market.close_time}")
        if market.close_time:
            current_time = datetime.now(timezone.utc)
            logger.info(f"Market was closed {current_time - market.close_time}.")

            agent.challenge_market(market, api_keys)
        else:
            logger.info("Market has not been closed yet.")


if __name__ == "__main__":
    typer.run(main)
