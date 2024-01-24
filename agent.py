import typer
import time
import logging
from decimal import Decimal

import prediction_market_agent as pma
from prediction_market_agent.markets.all_markets import (
    MarketType,
    get_binary_markets,
    omen,
    manifold,
)
from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.agents.all_agents import AgentType, get_agent
from prediction_market_agent.tools.utils import should_not_happen, check_not_none
from prediction_market_agent.tools.gtypes import Mana, xDai


def main(
    market_type: MarketType = MarketType.MANIFOLD,
    agent_type: AgentType = AgentType.ALWAYS_YES,
    sleep_time: int = 1 * 24 * 60 * 60,
) -> None:
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

        # TODO: Pick the market with agent-specific logic, for now just pick the first one.
        market = available_markets[0]  # agent.pick_market(available_markets)
        logging.info(f"Picked market [{market.id}]: {market.question}")
        answer = agent.answer_boolean_market(market)
        logging.info(f"Answered market [{market.id}]: {answer}")

        # TODO: Calculate the amount to bet based on the confidence of the answer.
        amount = Decimal("0.1")
        logging.info(f"Placing bet of {amount} on market [{market.id}]: {answer}")

        pma.manifold.place_bet(
            amount=Mana(amount),
            market_id=market.id,
            outcome=answer,
            api_key=check_not_none(keys.manifold),
        ) if isinstance(
            market, manifold.ManifoldMarket
        ) else pma.omen.binary_omen_buy_outcome_tx(
            amount=xDai(amount),
            from_address=check_not_none(keys.bet_from_address),
            from_private_key=check_not_none(keys.bet_from_private_key),
            market=market,
            binary_outcome=answer,
            auto_deposit=True,
        ) if isinstance(
            market, omen.OmenMarket
        ) else should_not_happen(
            f"Unknown market: {market}"
        )

        logging.info(f"Sleeping for {sleep_time} seconds...")
        time.sleep(sleep_time)


if __name__ == "__main__":
    typer.run(main)
