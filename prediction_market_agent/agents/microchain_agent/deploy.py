from microchain import Agent
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.microchain_agent import (
    build_agent,
    get_editable_prompt_from_agent,
)
from prediction_market_agent.agents.microchain_agent.prompts import (
    SYSTEM_PROMPTS,
    SystemPromptChoice,
)
from prediction_market_agent.agents.utils import AgentIdentifier
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler


class DeployableMicrochainAgent(DeployableAgent):
    model = "gpt-4o-2024-05-13"
    n_iterations = 50
    load_historical_prompt: bool = False
    system_prompt_choice: SystemPromptChoice = SystemPromptChoice.TRADING_AGENT

    def run(
        self,
        market_type: MarketType,
    ) -> None:
        """
        Override main 'run' method, as the all logic from the helper methods
        is handed over to the agent.
        """
        task_description = AgentIdentifier.microchain_task_from_market(market_type)
        long_term_memory = LongTermMemoryTableHandler(task_description=task_description)
        prompt_handler = PromptTableHandler(
            session_identifier=AgentIdentifier.MICROCHAIN_AGENT_OMEN
        )
        system_prompt = SYSTEM_PROMPTS[self.system_prompt_choice]
        agent: Agent = build_agent(
            market_type=market_type,
            model=self.model,
            system_prompt=system_prompt,
            allow_stop=True,
            long_term_memory=long_term_memory,
            prompt_handler=prompt_handler if self.load_historical_prompt else None,
        )
        agent.run(self.n_iterations)
        long_term_memory.save_history(agent.history)
        editable_prompt = get_editable_prompt_from_agent(agent)
        prompt_handler.save_prompt(editable_prompt)


class DeployableMicrochainModifiableSystemPromptAgent(DeployableMicrochainAgent):
    system_prompt_choice: SystemPromptChoice = SystemPromptChoice.JUST_BORN
    load_historical_prompt: bool = True
