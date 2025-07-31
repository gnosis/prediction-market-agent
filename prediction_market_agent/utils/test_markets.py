"""Test utilities for creating mock market objects for benchmarking and testing."""

from prediction_market_agent_tooling.markets.agent_market import AgentMarket


class TestAgentMarket(AgentMarket):
    """Simple test implementation of AgentMarket for benchmarking.
    
    This class provides a minimal concrete implementation of AgentMarket
    that can be used in tests and benchmarks without depending on any
    specific market platform.
    """
    
    base_url: str = "https://test.example.com"
    
    @property
    def is_multiresult(self) -> bool:
        return False
    
    def have_bet_on_market_since(self, keys, since):
        return False