from datetime import datetime, timezone

import typer
from eth_typing import HexAddress
from eth_typing.evm import HexStr
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.agents.ofvchallenger_agent.deploy import OFVChallengerAgent
from prediction_market_agent.utils import APIKeys

MARKET_IDS: list[str] = ["000000000000000000000000000000000000000000"]


def main(market_ids: list[str] = MARKET_IDS) -> None:
    api_keys = APIKeys()
    agent = OFVChallengerAgent()

    market_ids_hex: list[HexAddress] = [
        HexAddress(HexStr(market_id)) for market_id in market_ids
    ]

    markets = []
    for market_id in market_ids_hex:
        try:
            markets.append(
                OmenSubgraphHandler().get_omen_market_by_market_id(market_id)
            )
        except Exception as e:
            logger.error(f"Failed to retrieve market {market_id}: {e}")

    for market in markets:
        logger.info(
            f"Challenging market {market.question.id=} - {market.question.title=} - {market.question.outcomes=}"
        )
        logger.info(f"Market creation timestamp: {market.creation_datetime}")
        logger.info(f"Market close time: {market.close_time}")
        if market.close_time and not market.is_resolved_with_valid_answer:
            logger.info(f"Market was closed {utcnow() - market.close_time}.")

            try:
                agent.challenge_market(market, api_keys)
            except Exception as e:
                logger.error(f"Failed to challenge for {market.question}: {e}")
        else:
            logger.info("Market has not been closed yet.")


if __name__ == "__main__":
    typer.run(main)
