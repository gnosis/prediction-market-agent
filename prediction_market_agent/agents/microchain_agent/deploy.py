import abc
import time

from microchain import Agent
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import check_not_none

from prediction_market_agent.agents.goal_manager import Goal, GoalManager
from prediction_market_agent.agents.identifiers import AgentIdentifier
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
    JUST_BORN_SYSTEM_PROMPT_CONFIG,
    TRADING_AGENT_SYSTEM_PROMPT_CONFIG,
    TRADING_AGENT_SYSTEM_PROMPT_MINIMAL_CONFIG,
    FunctionsConfig,
)
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.prompt_table_handler import PromptTableHandler
from prediction_market_agent.utils import APIKeys

GENERAL_AGENT_TAG = "general_agent"


class DeployableMicrochainAgentAbstract(DeployableAgent, metaclass=abc.ABCMeta):
    model = SupportedModel.gpt_4o
    max_iterations: int | None = 50
    import_actions_from_memory = 0
    sleep_between_iterations = 0
    identifier: AgentIdentifier
    functions_config: FunctionsConfig

    @classmethod
    def get_description(cls) -> str:
        return f"Microchain-based {cls.__name__}."

    @classmethod
    @abc.abstractmethod
    def get_initial_system_prompt(cls) -> str:
        pass

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
        self.run_general_agent(market_type=market_type)

    @observe()
    def run_general_agent(
        self,
        market_type: MarketType,
    ) -> None:
        self.langfuse_update_current_trace(tags=[GENERAL_AGENT_TAG, self.identifier])

        long_term_memory = LongTermMemoryTableHandler.from_agent_identifier(
            self.identifier
        )
        prompt_handler = PromptTableHandler.from_agent_identifier(self.identifier)
        unformatted_system_prompt = get_unformatted_system_prompt(
            unformatted_prompt=self.get_initial_system_prompt(),
            prompt_table_handler=prompt_handler,
        )

        agent: Agent = build_agent(
            market_type=market_type,
            model=self.model,
            unformatted_system_prompt=unformatted_system_prompt,
            allow_stop=True,
            long_term_memory=long_term_memory,
            import_actions_from_memory=self.import_actions_from_memory,
            keys=APIKeys(),
            functions_config=self.functions_config,
            enable_langfuse=self.enable_langfuse,
        )

        goal_manager = self.build_goal_manager(agent=agent)
        goal = goal_manager.get_goal() if goal_manager else None
        if goal:
            agent.prompt = goal.to_prompt()

        # Save formatted system prompt
        initial_formatted_system_prompt = agent.system_prompt

        iteration = 0
        while self.max_iterations is None or iteration < self.max_iterations:
            starting_history_length = len(agent.history)
            try:
                # After the first iteration, resume=True to not re-initialize the agent.
                agent.run(iterations=1, resume=iteration > 0)
            except Exception as e:
                logger.error(f"Error while running microchain agent: {e}")
                raise e
            finally:
                # Save the agent's history to the long-term memory after every iteration to keep users updated.
                save_agent_history(
                    agent=agent,
                    long_term_memory=long_term_memory,
                    initial_system_prompt=initial_formatted_system_prompt,
                    # Because the agent is running in a while cycle, always save into database only what's new, to not duplicate entries.
                    save_last_n=len(agent.history) - starting_history_length,
                )
                if agent.system_prompt != initial_formatted_system_prompt:
                    prompt_handler.save_prompt(get_editable_prompt_from_agent(agent))
            iteration += 1
            logger.info(f"{self.__class__.__name__} iteration {iteration} completed.")
            if self.sleep_between_iterations:
                logger.info(
                    f"{self.__class__.__name__} sleeping for {self.sleep_between_iterations} seconds."
                )
                time.sleep(self.sleep_between_iterations)

        if goal_manager:
            self.handle_goal_evaluation(
                agent,
                check_not_none(goal),
                goal_manager,
                long_term_memory,
                initial_formatted_system_prompt,
            )

    def handle_goal_evaluation(
        self,
        agent: Agent,
        goal: Goal,
        goal_manager: GoalManager,
        long_term_memory: LongTermMemoryTableHandler,
        initial_formatted_system_prompt: str,
    ) -> None:
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
                content=f"# Goal evaluation\n{goal_evaluation}",
            ).model_dump()
        )
        save_agent_history(
            agent=agent,
            long_term_memory=long_term_memory,
            initial_system_prompt=initial_formatted_system_prompt,
            # Save only the new (last) message, which is the goal evaluation.
            save_last_n=1,
        )


class DeployableMicrochainAgent(DeployableMicrochainAgentAbstract):
    identifier = AgentIdentifier.MICROCHAIN_AGENT_OMEN
    functions_config = TRADING_AGENT_SYSTEM_PROMPT_CONFIG.functions_config

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        return TRADING_AGENT_SYSTEM_PROMPT_CONFIG.system_prompt


class DeployableMicrochainModifiableSystemPromptAgentAbstract(
    DeployableMicrochainAgent
):
    functions_config = JUST_BORN_SYSTEM_PROMPT_CONFIG.functions_config

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        return JUST_BORN_SYSTEM_PROMPT_CONFIG.system_prompt


class DeployableMicrochainModifiableSystemPromptAgent0(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    identifier = AgentIdentifier.MICROCHAIN_AGENT_OMEN_LEARNING_0

    @classmethod
    def get_description(cls) -> str:
        return "Microchain agent with 'just born' system prompt, and ability to adjust its own system prompt, version 0."


class DeployableMicrochainModifiableSystemPromptAgent1(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    identifier = AgentIdentifier.MICROCHAIN_AGENT_OMEN_LEARNING_1

    @classmethod
    def get_description(cls) -> str:
        return "Microchain agent with 'just born' system prompt, and ability to adjust its own system prompt, version 1."


class DeployableMicrochainModifiableSystemPromptAgent2(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    identifier = AgentIdentifier.MICROCHAIN_AGENT_OMEN_LEARNING_2

    @classmethod
    def get_description(cls) -> str:
        return "Microchain agent with 'just born' system prompt, and ability to adjust its own system prompt, version 2."


class DeployableMicrochainModifiableSystemPromptAgent3(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    identifier = AgentIdentifier.MICROCHAIN_AGENT_OMEN_LEARNING_3
    model = SupportedModel.llama_31_instruct
    # Force less iterations, because Replicate's API allows at max 4096 input tokens.
    max_iterations = 10

    @classmethod
    def get_description(cls) -> str:
        return "Microchain agent with 'just born' system prompt, and ability to adjust its own system prompt, version 3. Uses Llama 3.1 model."


class DeployableMicrochainWithGoalManagerAgent0(DeployableMicrochainAgent):
    identifier = AgentIdentifier.MICROCHAIN_AGENT_OMEN_WITH_GOAL_MANAGER
    model = SupportedModel.gpt_4o
    functions_config = TRADING_AGENT_SYSTEM_PROMPT_MINIMAL_CONFIG.functions_config

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        return TRADING_AGENT_SYSTEM_PROMPT_MINIMAL_CONFIG.system_prompt

    @classmethod
    def get_description(cls) -> str:
        return "Microchain agent woth minimal 'trader' system prompt, and GoalManager, version 0"

    def build_goal_manager(
        self,
        agent: Agent,
    ) -> GoalManager:
        return GoalManager(
            agent_id=self.identifier,
            high_level_description="You are a trader agent in prediction markets, aiming to maximise your long-term profit.",
            agent_capabilities=f"You have the following capabilities:\n{get_functions_summary_list(agent.engine)}",
            retry_limit=1,
            goal_history_limit=10,
        )
