import typer
import time
import logging
from decimal import Decimal
from datetime import timedelta

import prediction_market_agent as pma
from prediction_market_agent.markets.all_markets import (
    MarketType,
    get_binary_markets,
    place_bet,
)
from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.agents.all_agents import AgentType, get_agent


def main(
    market_type: MarketType = MarketType.MANIFOLD,
    agent_type: AgentType = AgentType.ALWAYS_YES,
    sleep_time: int = timedelta(days=1).seconds,
):
    """
    Start the agent as a continuous process. Picks a market and answers it, forever and ever.
    """
    agent: AbstractAgent = get_agent(agent_type)
    keys = pma.utils.get_keys()

    while True:
        # TODO: Agent needs to keep track of the questions it has answered. It should skip them or re-evaluate.
        available_markets = get_binary_markets(market_type)
        logging.info(
            f"Found {len(available_markets)} markets: {[m.question for m in available_markets]}"
        )

        market = agent.pick_market(available_markets)
        logging.info(f"Picked market [{market.id}]: {market.question}")
        answer = agent.answer_boolean_market(market)
        logging.info(f"Answered market [{market.id}]: {answer}")

        # TODO: Calculate the amount to bet based on the confidence of the answer.
        amount = Decimal("0.1")
        logging.info(f"Placing bet of {amount} on market [{market.id}]: {answer}")

        place_bet(
            market=market,
            amount=amount,
            outcome=answer,
            keys=keys,
            omen_auto_deposit=True,
        )

        logging.info(f"Sleeping for {timedelta(seconds=sleep_time)}...")
        time.sleep(sleep_time)


if __name__ == "__main__":
    typer.run(main)
