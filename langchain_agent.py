from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI

import manifold
from utils import (
    get_serp_api_key,
    get_openai_api_key,
    get_manifold_api_key,
    get_market_prompt,
    parse_result_to_boolean,
)

# Note: there is an experimental autogpt API
# from langchain_experimental.autonomous_agents.autogpt.agent import AutoGPT

llm = OpenAI(openai_api_key=get_openai_api_key())
tools = load_tools(["serpapi", "llm-math"], llm=llm, serpapi_api_key=get_serp_api_key())

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

market = manifold.pick_binary_market()
result = agent.run(get_market_prompt(market.question))
manifold.place_bet(
    amount=5,
    market_id=market.id,
    outcome=parse_result_to_boolean(result),
    api_key=get_manifold_api_key(),
)
