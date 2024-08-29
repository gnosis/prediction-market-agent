from microchain import Agent
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import check_not_none

from prediction_market_agent.agents.goal_manager import GoalManager
from prediction_market_agent.agents.microchain_agent.memory import (
    ChatHistory,
    ChatMessage,
)
from prediction_market_agent.agents.microchain_agent.microchain_agent import (
    SupportedModel,
    build_agent,
    get_editable_prompt_from_agent,
    get_functions_summary_list,
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

GENERAL_AGENT_TAG = "general_agent"


class DeployableMicrochainAgent(DeployableAgent):
    model = SupportedModel.gpt_4o
    n_iterations = 50
    load_historical_prompt: bool = False
    system_prompt_choice: SystemPromptChoice = SystemPromptChoice.TRADING_AGENT
    task_description = AgentIdentifier.MICROCHAIN_AGENT_OMEN

    def build_goal_manager(
        self,
        agent: Agent,
    ) -> GoalManager | None:
        return None

    def run(
        self,
        market_type: MarketType,
    ) -> None:
        """
        Override main 'run' method, as the all logic from the helper methods
        is handed over to the agent.
        """
        self.langfuse_update_current_trace(
            tags=[GENERAL_AGENT_TAG, self.system_prompt_choice, self.task_description]
        )

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
            enable_langfuse=self.enable_langfuse,
        )

        if goal_manager := self.build_goal_manager(agent=agent):
            goal = goal_manager.get_goal()
            agent.prompt = goal.to_prompt()

        # Save formatted system prompt
        initial_formatted_system_prompt = agent.system_prompt

        try:
            agent.run(self.n_iterations)
        except Exception as e:
            logger.error(e)
            raise e
        finally:
            if goal_manager:
                goal = check_not_none(goal)
                goal_evaluation = goal_manager.evaluate_goal_progress(
                    goal=goal,
                    chat_history=ChatHistory.from_list_of_dicts(agent.history),
                )
                goal_manager.save_evaluated_goal(
                    goal=goal,
                    evaluation=goal_evaluation,
                )
                agent.history.append(
                    ChatMessage(
                        role="user",
                        content=str(f"# Goal evaluation\n{goal_evaluation}"),
                    ).model_dump()
                )

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


class DeployableMicrochainWithGoalManagerAgent0(DeployableMicrochainAgent):
    task_description = AgentIdentifier.MICROCHAIN_AGENT_OMEN_WITH_GOAL_MANAGER
    model = SupportedModel.gpt_4o
    system_prompt_choice = SystemPromptChoice.TRADING_AGENT_MINIMAL

    def build_goal_manager(
        self,
        agent: Agent,
    ) -> GoalManager:
        return GoalManager(
            agent_id=self.task_description,
            high_level_description="You are a trader agent in prediction markets, aiming to maximise your long-term profit.",
            agent_capabilities=f"You have the following capabilities:\n{get_functions_summary_list(agent.engine)}",
            retry_limit=3,
        )
