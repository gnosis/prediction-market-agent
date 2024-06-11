from microchain import Agent
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.memory import LongTermMemory
from prediction_market_agent.agents.microchain_agent.microchain_agent import build_agent
from prediction_market_agent.agents.microchain_agent.prompts import (
    TRADING_AGENT_BOOTSTRAP,
    TRADING_AGENT_SYSTEM_PROMPT,
)
from prediction_market_agent.agents.microchain_agent.prompt_handler import PromptHandler
from prediction_market_agent.agents.microchain_agent.prompts import (
    TRADING_AGENT_BOOTSTRAP,
    TRADING_AGENT_SYSTEM_PROMPT,
)
from prediction_market_agent.agents.utils import LongTermMemoryTaskIdentifier


class DeployableMicrochainAgent(DeployableAgent):
    model = "gpt-4o-2024-05-13"
    n_iterations = 50
    load_historical_prompt: bool = False

    def run(self, market_type: MarketType) -> None:
        """
        Override main 'run' method, as the all logic from the helper methods
        is handed over to the agent.
        """
        task_description = LongTermMemoryTaskIdentifier.microchain_task_from_market(
            market_type
        )
        long_term_memory = LongTermMemory(task_description=task_description)
        prompt_handler = PromptHandler()
        agent: Agent = build_agent(
            market_type=market_type,
            model=self.model,
            system_prompt=TRADING_AGENT_SYSTEM_PROMPT,  # Use pre-learned system prompt until the prompt-fetching from DB is implemented.
            bootstrap=TRADING_AGENT_BOOTSTRAP,  # Same here.
            allow_stop=True,
            long_term_memory=long_term_memory,
            prompt_handler=prompt_handler if self.load_historical_prompt else None,
        )
        agent.run(self.n_iterations)
        long_term_memory.save_history(agent.history)
        prompt_handler.save_prompt(agent.system_prompt)
