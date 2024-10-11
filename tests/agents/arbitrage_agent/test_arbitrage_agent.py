import typing as t
from unittest.mock import Mock, patch

import pytest
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket

from prediction_market_agent.agents.arbitrage_agent.deploy import (
    DeployableArbitrageAgent,
)


@pytest.fixture(scope="module")
def arbitrage_agent() -> t.Generator[DeployableArbitrageAgent, None, None]:
    with patch(
        "prediction_market_agent.agents.arbitrage_agent.deploy.DeployableArbitrageAgent.load",
        new=lambda x: None,
    ), patch(
        "prediction_market_agent_tooling.tools.langfuse_.get_langfuse_langchain_config"
    ):
        agent = DeployableArbitrageAgent()
        # needed since load was mocked
        agent.chain = agent._build_chain()
        yield agent


@pytest.fixture(scope="module")
def main_market() -> t.Generator[AgentMarket, None, None]:
    m1 = Mock(OmenAgentMarket, wraps=OmenAgentMarket)
    m1.question = "Will Kamala Harris win the US presidential election in 2024?"
    yield m1


@pytest.fixture(scope="module")
def related_market() -> t.Generator[AgentMarket, None, None]:
    m1 = Mock(OmenAgentMarket, wraps=OmenAgentMarket)
    m1.question = "Will Kamala Harris become the US president in 2025?"
    yield m1


@pytest.fixture(scope="module")
def unrelated_market() -> t.Generator[AgentMarket, None, None]:
    m1 = Mock(OmenAgentMarket, wraps=OmenAgentMarket)
    m1.question = "Will Donald Duck ever retire from his adventures in Duckburg?"
    yield m1


@pytest.mark.parametrize(
    "related_market_fixture_name, is_correlated",
    [("related_market", True), ("unrelated_market", False)],
)
def test_correlation_for_similar_markets(
    arbitrage_agent: DeployableArbitrageAgent,
    main_market: AgentMarket,
    related_market_fixture_name: str,
    is_correlated: bool,
    request: pytest.FixtureRequest,
) -> None:
    other_market = request.getfixturevalue(related_market_fixture_name)
    correlation = arbitrage_agent.calculate_correlation_between_markets(
        market=main_market, related_market=other_market
    )
    assert correlation.near_perfect_correlation == is_correlated
