"""
PYTHONPATH=. streamlit run scripts/agent_app.py

Tip: if you specify PYTHONPATH=., streamlit will watch for the changes in all files, instead of just this one.
"""

import streamlit as st

st.set_page_config(layout="wide")

import nest_asyncio

nest_asyncio.apply()  # Required for streamlit to work with asyncio.

from prediction_market_agent.utils import patch_sqlite3

patch_sqlite3()
import gc
import time
import typing as t

from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    QuestionType,
)
from prediction_market_agent_tooling.markets.markets import (
    MarketType,
    get_binary_markets,
)
from prediction_market_agent_tooling.tools.streamlit_user_login import streamlit_login

from prediction_market_agent.agents.known_outcome_agent.deploy import (
    DeployableKnownOutcomeAgent,
)
from prediction_market_agent.agents.prophet_agent.deploy import (
    DeployableOlasEmbeddingOAAgent,
    DeployablePredictionProphetGPT4oAgent,
    DeployablePredictionProphetGPT4oAgentScalar,
    DeployablePredictionProphetGPT4TurboFinalAgent,
    DeployablePredictionProphetGPT4TurboPreviewAgent,
    DeployablePredictionProphetGPTo1MiniAgent,
    DeployablePredictionProphetGPTo1PreviewAgent,
)
from prediction_market_agent.agents.think_thoroughly_agent.deploy import (
    DeployableThinkThoroughlyAgent,
    DeployableThinkThoroughlyProphetResearchAgent,
)
from prediction_market_agent.tools.streamlit_utils import (
    add_sink_to_logger,
    streamlit_escape,
)

SupportedAgentType: t.TypeAlias = (
    type[DeployableKnownOutcomeAgent]
    | type[DeployableThinkThoroughlyAgent]
    | type[DeployableThinkThoroughlyProphetResearchAgent]
    | type[DeployablePredictionProphetGPT4oAgent]
    | type[DeployablePredictionProphetGPT4TurboPreviewAgent]
    | type[DeployablePredictionProphetGPT4TurboFinalAgent]
    | type[DeployableOlasEmbeddingOAAgent]
    | type[DeployablePredictionProphetGPT4oAgentScalar]
    | type[DeployablePredictionProphetGPTo1PreviewAgent]
    | type[DeployablePredictionProphetGPTo1MiniAgent]
)

AGENTS: list[SupportedAgentType] = [
    DeployableKnownOutcomeAgent,
    DeployableThinkThoroughlyAgent,
    DeployableThinkThoroughlyProphetResearchAgent,
    DeployablePredictionProphetGPT4oAgent,
    DeployablePredictionProphetGPT4TurboPreviewAgent,
    DeployablePredictionProphetGPT4TurboFinalAgent,
    DeployableOlasEmbeddingOAAgent,
    DeployablePredictionProphetGPT4oAgentScalar,
    DeployablePredictionProphetGPTo1PreviewAgent,
    DeployablePredictionProphetGPTo1MiniAgent,
]

add_sink_to_logger()


def predict(
    AgentClass: SupportedAgentType,
    market_source: MarketType,
    market: AgentMarket,
    skip_market_verification: bool,
    enable_langfuse: bool,
) -> None:
    agent = AgentClass(
        place_trades=False,
        store_predictions=False,
        store_trades=False,
        enable_langfuse=enable_langfuse,
    )

    start_time = time.time()

    # Show the agent's title.
    st.write(
        f"## {agent.__class__.__name__.replace('Deployable', '').replace('Agent', '')}"
    )

    with st.spinner("Agent is processing the market..."):
        processed_market = agent.process_market(
            market_source,
            market,
            verify_market=not skip_market_verification,
        )

    if processed_market is None:
        st.error("Agent failed to process this market.")
        return

    end_time = time.time() - start_time
    st.warning(f"Took {end_time / 60:.2f} minutes.")
    st.success(
        streamlit_escape(
            f"Would do the following trades {processed_market.trades} on {processed_market.answer}!"
        )
    )


def agent_app() -> None:
    st.title("Agent's decision-making process")

    with st.sidebar:
        streamlit_login()

    # Fetch markets from the selected market type.
    market_source = MarketType(
        st.selectbox(
            "Select a market source",
            [market_source.value for market_source in MarketType],
        )
    )
    fetch_scalar_markets = st.checkbox("Fetch scalar markets", value=False)
    markets = get_binary_markets(
        42,
        market_source,
        question_type=(
            QuestionType.SCALAR if fetch_scalar_markets else QuestionType.BINARY
        ),
    )

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

    if not custom_question_input and market.is_binary:
        st.info(f"Current probability {market.p_yes * 100:.2f}% at {market.url}.")
    elif not custom_question_input and market.is_scalar:
        st.info(f"Current UP probability {market.p_up * 100:.2f}% at {market.url}.")
    else:
        st.info(f"Current question {market.question} at {market.url}.")

    skip_market_verification = st.checkbox(
        "Skip market verification", value=False, key="skip_market_verification"
    )

    # Select an agent from the list of available agents.
    agent_class_names = st.multiselect(
        "Select agents", [agent_class.__name__ for agent_class in AGENTS]
    )
    if not agent_class_names:
        st.warning("Please select at least one agent.")
        st.stop()

    # Get the agent classes from the names.
    agent_classes: list[SupportedAgentType] = []
    for AgentClass in AGENTS:
        if AgentClass.__name__ in agent_class_names:
            agent_classes.append(AgentClass)

    for column, AgentClass in zip(st.columns(len(agent_classes)), agent_classes):
        with column:
            try:
                predict(
                    AgentClass=AgentClass,
                    market_source=market_source,
                    market=market,
                    skip_market_verification=skip_market_verification,
                    enable_langfuse=APIKeys().default_enable_langfuse
                    and not custom_question_input  # Don't store custom inputs, as we won't have true data for them.
                    and not skip_market_verification,  # Don't store unverified markets, as they wouldn't be predicted anyway.
                )
            finally:
                # Force memory cleanup no matter what.
                gc.collect()


if __name__ == "__main__":
    agent_app()
