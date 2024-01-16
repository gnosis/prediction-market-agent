from enum import Enum
import argparse

import prediction_market_agent as pma


class AgentType(Enum):
    LANGCHAIN = 1
    AUTOGEN = 2
    ALWAYS_YES = 3
    LLAMAINDEX = 4
    METAGPT = 5
    CREWAI = 6


class Market(Enum):
    MANIFOLD = 1
    OMEN = 2


agent_mapping = {
    AgentType.LANGCHAIN: pma.agents.langchain.LangChainAgent,
    AgentType.AUTOGEN: pma.agents.autogen.AutoGenAgent,
    AgentType.ALWAYS_YES: pma.agents.always_yes.AlwaysYesAgent,
    AgentType.LLAMAINDEX: pma.agents.llamaindex.LlamaIndexAgent,
    AgentType.METAGPT: pma.agents.metagpt.MetaGPTAgent,
    AgentType.CREWAI: pma.agents.crewai.CrewAIAgent,
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
        default="always_yes",
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

    if market != Market.MANIFOLD:
        raise NotImplementedError(
            "Only Manifold is supported for betting at the moment."
        )

    if do_bet:
        print(
            f"Placing bet with position {pma.utils.parse_result_to_str(result)} on market '{market.question}'"
        )
        pma.manifold.place_bet(
            amount=5,
            market_id=market.id,
            outcome=result,
            api_key=keys.manifold,
        )
