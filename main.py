from enum import Enum
import argparse

import prediction_market_agent as pma
from prediction_market_agent.tools.types import xdai_type
from prediction_market_agent.tools.utils import should_not_happen, check_not_none


class AgentType(Enum):
    LANGCHAIN = 1
    AUTOGEN = 2
    ALWAYS_YES = 3
    LLAMAINDEX = 4
    METAGPT = 5
    CREWAI = 6
    CUSTOM_OPENAI = 7
    CUSTOM_LLAMA = 8


class Market(Enum):
    MANIFOLD = 1
    OMEN = 2


agent_mapping = {
    AgentType.LANGCHAIN: pma.agents.langchain_agent.LangChainAgent,
    AgentType.AUTOGEN: pma.agents.autogen_agent.AutoGenAgent,
    AgentType.ALWAYS_YES: pma.agents.always_yes.AlwaysYesAgent,
    AgentType.LLAMAINDEX: pma.agents.llamaindex_agent.LlamaIndexAgent,
    AgentType.METAGPT: pma.agents.metagpt_agent.MetaGPTAgent,
    AgentType.CREWAI: pma.agents.crewai_agent.CrewAIAgent,
    AgentType.CUSTOM_OPENAI: pma.agents.custom_agent.CustomAgent.init_with_openai,
    AgentType.CUSTOM_LLAMA: pma.agents.custom_agent.CustomAgent.init_with_llama,
}

pick_binary_market_mapping = {
    Market.MANIFOLD: pma.manifold.pick_binary_market,
    Market.OMEN: pma.omen.pick_binary_market,
}

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--agent-type",
        type=str,
        choices=[t.name.lower() for t in list(AgentType)],
        default=AgentType.ALWAYS_YES.name.lower(),
    )
    args.add_argument(
        "--market",
        type=str,
        choices=[t.name.lower() for t in list(Market)],
        default=Market.MANIFOLD.name.lower(),
    )
    args.add_argument(
        "--auto-bet",
        type=bool,
        default=False,
        help="If true, does not require user input to place the bet.",
    )

    parsed_args = args.parse_args()
    selected_market = Market[parsed_args.market.upper()]
    agent_type = AgentType[parsed_args.agent_type.upper()]
    keys = pma.utils.get_keys()

    # Pick a market
    market = pick_binary_market_mapping[selected_market]()

    # Create the agent and run it
    agent = agent_mapping[agent_type]()
    result = agent.run(market.question)

    # Place a bet based on the result
    if parsed_args.auto_bet:
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
        amount = xdai_type(input("How much do you want to bet? (in xDai): "))
        pma.manifold.place_bet(
            amount=amount,
            market_id=market.id,
            outcome=result,
            api_key=keys.manifold,
        ) if selected_market == Market.MANIFOLD else pma.omen.binary_omen_buy_outcome_tx(
            amount=amount,
            from_address=check_not_none(keys.bet_from_address),
            from_private_key=check_not_none(keys.bet_from_private_key),
            market=market,
            binary_outcome=result,
            auto_deposit=True,
        ) if selected_market == Market.OMEN else should_not_happen(
            f"Unknown market: {market}"
        )
