"""
PYTHONPATH=. streamlit run scripts/agent_app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, isntead of just this one.
"""

import streamlit as st

st.set_page_config(layout="wide")

from prediction_market_agent.utils import patch_sqlite3

patch_sqlite3()

import typing as t

from prediction_market_agent_tooling.markets.markets import (
    MarketType,
    get_binary_markets,
)
from prediction_market_agent_tooling.tools.costs import openai_costs

from prediction_market_agent.agents.known_outcome_agent.deploy import (
    DeployableKnownOutcomeAgent,
)
from prediction_market_agent.agents.think_thoroughly_agent.deploy import (
    DeployableThinkThoroughlyAgent,
)
from prediction_market_agent.tools.streamlit_utils import (
    add_sink_to_logger,
    streamlit_escape,
)

AGENTS: list[
    t.Type[DeployableKnownOutcomeAgent] | t.Type[DeployableThinkThoroughlyAgent]
] = [DeployableKnownOutcomeAgent, DeployableThinkThoroughlyAgent]

add_sink_to_logger()

st.title("Agent's decision-making process")

# Fetch markets from the selected market type.
market_source = MarketType(
    st.selectbox(
        "Select a market source", [market_source.value for market_source in MarketType]
    )
)
markets = get_binary_markets(42, market_source)

# Select an agent from the list of available agents.
agent_class_names = st.multiselect(
    "Select agents", [agent_class.__name__ for agent_class in AGENTS]
)
if not agent_class_names:
    st.warning("Please select at least one agent.")
    st.stop()

# Get the agent classes from the names.
agent_classes: list[
    t.Type[DeployableKnownOutcomeAgent] | t.Type[DeployableThinkThoroughlyAgent]
] = []
for AgentClass in AGENTS:
    if AgentClass.__name__ in agent_class_names:
        agent_classes.append(AgentClass)

# Ask the user to provide a question.
custom_question_input = st.checkbox("Provide a custom question", value=False)
question = (
    st.text_input("Question")
    if custom_question_input
    else st.selectbox("Select a question", [m.question for m in markets])
)
if not question:
    st.warning("Please enter a question.")
    st.stop()

market = (
    [m for m in markets if m.question == question][0]
    if not custom_question_input
    # If custom question is provided, just take some random market and update its question.
    else markets[0].model_copy(update={"question": question, "current_p_yes": 0.5})
)

for idx, (column, AgentClass) in enumerate(
    zip(st.columns(len(agent_classes)), agent_classes)
):
    with column:
        agent = AgentClass()

        # This needs to be a separate block to measure the time and cost and then write it into the column section.
        with openai_costs(agent.model) as costs:
            # Show the agent's title.
            st.write(
                f"## {agent.__class__.__name__.replace('Deployable', '').replace('Agent', '')}"
            )

            # Simulate deployable agent logic.
            with st.spinner("Agent is verifying the market..."):
                if not agent.pick_markets([market]):
                    st.warning("Agent wouldn't pick this market to bet on.")
                    if not st.checkbox(
                        "Continue with the prediction anyway",
                        value=False,
                        key=f"continue_{idx}",
                    ):
                        continue

            with st.spinner("Agent is making a decision..."):
                answer = agent.answer_binary_market(market)

            if answer is None:
                st.error("Agent failed to answer this market.")
                continue

            with st.spinner("Agent is calculating the bet amount..."):
                bet_amount = agent.calculate_bet_amount(answer, market)

        st.warning(f"Took {costs.time / 60:.2f} minutes and {costs.cost:.2f} USD.")
        st.success(
            streamlit_escape(
                f"Would bet {bet_amount.amount} {bet_amount.currency} on {answer}!"
            )
        )
