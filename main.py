from enum import Enum
import argparse

import utils
import manifold

from langchain_agent import LangChainAgent
from autogen_agent import AutoGenAgent
from always_yes_agent import AlwaysYesAgent


class AgentType(Enum):
    LANGCHAIN = 1
    AUTOGEN = 2
    ALWAYS_YES = 3


agent_mapping = {
    AgentType.LANGCHAIN: LangChainAgent,
    AgentType.AUTOGEN: AutoGenAgent,
    AgentType.ALWAYS_YES: AlwaysYesAgent,
}

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--agent-type",
        type=str,
        choices=[t.name.lower() for t in list(AgentType)],
        default="always_yes",
    )
    args.add_argument(
        "--auto-bet",
        type=bool,
        default=False,
        help="If true, does not require user input to place the bet.",
    )

    args = args.parse_args()
    agent_type = AgentType[args.agent_type.upper()]
    keys = utils.get_keys()

    # Pick a market
    market = manifold.pick_binary_market()

    # Create the agent and run it
    agent = agent_mapping[agent_type]()
    result = agent.run(utils.get_market_prompt(market.question))

    # Place a bet based on the result

    if args.auto_bet:
        do_bet = True
    else:
        prompt = (
            f"Do you want to take the position:\n\n{result.upper()}\n\n"
            f"on the market:\n\n{market.question}\n\n"
            f"(y/n, press Enter for default 'y'): "
        )
        user_input = input(prompt)
        do_bet = user_input.lower().strip() == "y" if user_input else True

    if do_bet:
        print(f"Placing bet with position {result} on market '{market.question}'")
        manifold.place_bet(
            amount=5,
            market_id=market.id,
            outcome=utils.parse_result_to_boolean(result),
            api_key=keys.manifold,
        )
