from microchain import Agent
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.agents.microchain_agent.microchain_agent import (
    SupportedModel,
    build_agent,
    get_editable_prompt_from_agent,
    get_unformatted_system_prompt,
    save_agent_history,
)
from prediction_market_agent.agents.microchain_agent.prompts import (
    SYSTEM_PROMPTS,
    FunctionsConfig,
    SystemPromptChoice,
)
from prediction_market_agent.agents.utils import AgentIdentifier
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler
from prediction_market_agent.utils import APIKeys


class DeployableMicrochainAgent(DeployableAgent):
    model = SupportedModel.gpt_4o
    n_iterations = 50
    load_historical_prompt: bool = False
    system_prompt_choice: SystemPromptChoice = SystemPromptChoice.TRADING_AGENT
    task_description = AgentIdentifier.MICROCHAIN_AGENT_OMEN

    def run(
        self,
        market_type: MarketType,
    ) -> None:
        """
        Override main 'run' method, as the all logic from the helper methods
        is handed over to the agent.
        """
        long_term_memory = LongTermMemoryTableHandler(
            task_description=self.task_description
        )
        prompt_handler = PromptTableHandler(session_identifier=self.task_description)
        unformatted_system_prompt = get_unformatted_system_prompt(
            unformatted_prompt=SYSTEM_PROMPTS[self.system_prompt_choice],
            prompt_table_handler=(
                prompt_handler if self.load_historical_prompt else None
            ),
        )
        agent: Agent = build_agent(
            market_type=market_type,
            model=self.model,
            unformatted_system_prompt=unformatted_system_prompt,
            allow_stop=True,
            long_term_memory=long_term_memory,
            keys=APIKeys(),
            functions_config=FunctionsConfig.from_system_prompt_choice(
                self.system_prompt_choice
            ),
        )

        # Save formatted system prompt
        initial_formatted_system_prompt = agent.system_prompt

        try:
            agent.run(self.n_iterations)
        except Exception as e:
            raise e
        finally:
            save_agent_history(
                agent=agent,
                long_term_memory=long_term_memory,
                initial_system_prompt=initial_formatted_system_prompt,
            )
            if agent.system_prompt != initial_formatted_system_prompt:
                prompt_handler.save_prompt(get_editable_prompt_from_agent(agent))


class DeployableMicrochainModifiableSystemPromptAgentAbstract(
    DeployableMicrochainAgent
):
    system_prompt_choice: SystemPromptChoice = SystemPromptChoice.JUST_BORN
    load_historical_prompt: bool = True
    task_description: AgentIdentifier


class DeployableMicrochainModifiableSystemPromptAgent0(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    task_description = AgentIdentifier.MICROCHAIN_AGENT_OMEN_LEARNING_0


class DeployableMicrochainModifiableSystemPromptAgent1(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    task_description = AgentIdentifier.MICROCHAIN_AGENT_OMEN_LEARNING_1


class DeployableMicrochainModifiableSystemPromptAgent2(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    task_description = AgentIdentifier.MICROCHAIN_AGENT_OMEN_LEARNING_2


class DeployableMicrochainModifiableSystemPromptAgent3(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    task_description = AgentIdentifier.MICROCHAIN_AGENT_OMEN_LEARNING_3
    model = SupportedModel.llama_31_instruct
