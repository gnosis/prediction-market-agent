import typer
import prediction_market_agent as pma
from decimal import Decimal
from prediction_market_agent.agents.all_agents import AgentType, get_agent
from prediction_market_agent_tooling.markets.markets import (
    MarketType,
    get_bet_amount,
    get_binary_markets,
    place_bet,
)


def main(
    market_type: MarketType = MarketType.MANIFOLD,
    agent_type: AgentType = AgentType.ALWAYS_YES,
    auto_bet: bool = False,
) -> None:
    """
    Picks one market and answers it, optionally placing a bet.
    """
    keys = pma.utils.APIKeys()

    # Pick a market
    market = get_binary_markets(market_type)[0].to_agent_market()

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
        print(
            f"Placing bet with position {pma.utils.parse_result_to_str(result)} on market '{market.question}'"
        )
        amount = Decimal(
            input(f"How much do you want to bet? (in {market.bet_amount_currency}): ")
        )
        place_bet(
            market=market.original_market,
            amount=get_bet_amount(amount, market_type),
            outcome=result,
            omen_auto_deposit=True,
        )


if __name__ == "__main__":
    typer.run(main)
