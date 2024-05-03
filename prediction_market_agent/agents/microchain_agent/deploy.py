from microchain import Agent
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.microchain_agent import get_agent


class DeployableMicrochainAgent(DeployableAgent):
    model = "gpt-4-1106-preview"
    n_iterations = 50

    def run(self, market_type: MarketType, _place_bet: bool = True) -> None:
        """
        Override main 'run' method, as the all logic from the helper methods
        is handed over to the agent.
        """
        agent: Agent = get_agent(
            market_type=market_type,
            model=self.model,
        )
        agent.bootstrap = ['Reasoning("I need to reason step by step")']
        agent.run(self.n_iterations)
