import typer
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import SortBy
from prediction_market_agent_tooling.markets.markets import (
    MarketType,
    get_binary_markets,
)

import prediction_market_agent as pma
from prediction_market_agent.agents.all_agents import AgentType, get_agent


def main(
    market_type: MarketType = MarketType.MANIFOLD,
    agent_type: AgentType = AgentType.ALWAYS_YES,
    auto_bet: bool = False,
) -> None:
    """
    Picks one market and answers it, optionally placing a bet.
    """
    # Pick a market
    market = get_binary_markets(
        limit=1,
        sort_by=SortBy.NEWEST,
        market_type=market_type,
    )[0]

    # Create the agent and run it
    agent = get_agent(agent_type)
    result = agent.answer_binary_market(market)

    # Place a bet based on the result
    if auto_bet:
        do_bet = True
    else:
        prompt = (
            f"Do you want to take the position:\n\n{pma.utils.parse_result_to_str(result)}\n\n"
            f"on the market:\n\n{market.question}\n\n"
            f"(y/n, press Enter for default 'y'): "
        )
        user_input = input(prompt)
        do_bet = user_input.lower().strip() == "y" if user_input else True

    if do_bet:
        logger.info(
            f"Placing bet with position {pma.utils.parse_result_to_str(result)} on market '{market.question}'"
        )
        amount = float(input(f"How much do you want to bet? (in {market.currency}): "))
        market.place_bet(
            amount=market.get_bet_amount(amount),
            outcome=result,
        )


if __name__ == "__main__":
    typer.run(main)
