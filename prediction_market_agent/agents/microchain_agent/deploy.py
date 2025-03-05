import abc
import time
from enum import Enum

from microchain import Agent
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import check_not_none

from prediction_market_agent.agents.goal_manager import Goal, GoalManager
from prediction_market_agent.agents.identifiers import (
    MICROCHAIN_AGENT_OMEN,
    MICROCHAIN_AGENT_OMEN_LEARNING_0,
    MICROCHAIN_AGENT_OMEN_LEARNING_1,
    MICROCHAIN_AGENT_OMEN_LEARNING_2,
    MICROCHAIN_AGENT_OMEN_LEARNING_3,
    MICROCHAIN_AGENT_OMEN_WITH_GOAL_MANAGER,
    AgentIdentifier,
)
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


class CallbackReturn(Enum):
    CONTINUE = "continue"
    STOP = "stop"


class DeployableMicrochainAgentAbstract(DeployableAgent, metaclass=abc.ABCMeta):
    # Setup per-agent class.
    model = SupportedModel.gpt_4o
    max_iterations: int | None = 50
    import_actions_from_memory = 0
    sleep_between_iterations = 0
    allow_stop: bool = True
    identifier: AgentIdentifier
    functions_config: FunctionsConfig
    initial_system_prompt: str

    # Setup during the 'load' method.
    long_term_memory: LongTermMemoryTableHandler
    prompt_handler: PromptTableHandler
    agent: Agent
    goal_manager: GoalManager | None
    api_keys: APIKeys

    @classmethod
    def get_description(cls) -> str:
        return f"Microchain-based {cls.__name__}."

    def build_long_term_memory(self) -> LongTermMemoryTableHandler:
        return LongTermMemoryTableHandler.from_agent_identifier(self.identifier)

    def build_prompt_handler(self) -> PromptTableHandler:
        return PromptTableHandler.from_agent_identifier(self.identifier)

    def build_agent(self, market_type: MarketType) -> Agent:
        unformatted_system_prompt = get_unformatted_system_prompt(
            unformatted_prompt=self.initial_system_prompt,
            prompt_table_handler=self.prompt_handler,
        )

        return build_agent(
            market_type=market_type,
            model=self.model,
            unformatted_system_prompt=unformatted_system_prompt,
            allow_stop=self.allow_stop,
            long_term_memory=self.long_term_memory,
            keys=APIKeys(),
            functions_config=self.functions_config,
            enable_langfuse=self.enable_langfuse,
        )

    def build_goal_manager(
        self,
        agent: Agent,
    ) -> GoalManager | None:
        return None

    def load(self) -> None:
        self.long_term_memory = self.build_long_term_memory()
        self.prompt_handler = self.build_prompt_handler()
        self.agent = self.build_agent(market_type=MarketType.OMEN)
        self.goal_manager = self.build_goal_manager(agent=self.agent)
        self.api_keys = APIKeys()

    def run(
        self,
        market_type: MarketType,
    ) -> None:
        """
        Override main 'run' method, as the all logic from the helper methods
        is handed over to the agent.
        """
        try:
            self.run_general_agent(market_type=market_type)
        finally:
            self.deinitialise_agent()

    def initialise_agent(self) -> None:
        logger.info(f"Initialising agent {self.__class__.__name__}.")
        self.agent.reset()
        self.agent.build_initial_messages()

        # Inject past history if wanted.
        if self.import_actions_from_memory:
            latest_saved_memories = self.long_term_memory.search(
                limit=self.import_actions_from_memory
            )
            messages_to_insert = [
                m.metadata_dict
                for m in latest_saved_memories[
                    ::-1
                ]  # Revert the list to have the oldest messages first, as they were in the history.
                if check_not_none(m.metadata_dict)["role"]
                != "system"  # Do not include system message as that one is automatically in the beginning of the history.
            ]
            # Inject them after the system message.
            self.agent.history[1:1] = messages_to_insert

    def deinitialise_agent(self) -> None:
        logger.info(f"Denitialising agent {self.__class__.__name__}.")

    @observe()
    def run_general_agent(self, market_type: MarketType) -> None:
        if market_type != MarketType.OMEN:
            raise ValueError(f"Only {MarketType.OMEN} market type is supported.")

        self.langfuse_update_current_trace(tags=[GENERAL_AGENT_TAG, self.identifier])

        goal = self.goal_manager.get_goal() if self.goal_manager else None
        if goal:
            self.agent.prompt = goal.to_prompt()

        # Initialise the agent before our working loop.
        self.initialise_agent()

        # Save formatted system prompt
        initial_formatted_system_prompt = self.agent.system_prompt

        iteration = 0

        while not self.agent.do_stop and (
            self.max_iterations is None or iteration < self.max_iterations
        ):
            if self.before_iteration_callback() == CallbackReturn.STOP:
                break

            starting_history_length = len(self.agent.history)
            try:
                # We initialise agent manually because of inserting past history, so force resume=True, to not re-initialise it which would remove the history.
                self.agent.run(iterations=1, resume=True)
            except Exception as e:
                logger.error(f"Error while running microchain agent: {e}")
                raise e
            finally:
                # Save the agent's history to the long-term memory after every iteration to keep users updated.
                self.save_agent_history(
                    initial_formatted_system_prompt=initial_formatted_system_prompt,
                    save_last_n=len(self.agent.history) - starting_history_length,
                )
                if self.agent.system_prompt != initial_formatted_system_prompt:
                    self.prompt_handler.save_prompt(
                        get_editable_prompt_from_agent(self.agent)
                    )

            iteration += 1
            logger.info(f"{self.__class__.__name__} iteration {iteration} completed.")

            if self.after_iteration_callback() == CallbackReturn.STOP:
                break

            if self.sleep_between_iterations:
                logger.info(
                    f"{self.__class__.__name__} sleeping for {self.sleep_between_iterations} seconds."
                )
                time.sleep(self.sleep_between_iterations)

        if self.goal_manager:
            self.handle_goal_evaluation(
                check_not_none(goal), initial_formatted_system_prompt
            )

    def save_agent_history(
        self, initial_formatted_system_prompt: str, save_last_n: int
    ) -> None:
        save_agent_history(
            agent=self.agent,
            long_term_memory=self.long_term_memory,
            initial_system_prompt=initial_formatted_system_prompt,
            # Because the agent is running in a while cycle, always save into database only what's new, to not duplicate entries.
            save_last_n=save_last_n,
        )

    def before_iteration_callback(self) -> CallbackReturn:
        return CallbackReturn.CONTINUE

    def after_iteration_callback(self) -> CallbackReturn:
        return CallbackReturn.CONTINUE

    def handle_goal_evaluation(
        self,
        goal: Goal,
        initial_formatted_system_prompt: str,
    ) -> None:
        assert self.goal_manager is not None, "Goal manager must be set."
        goal_evaluation = self.goal_manager.evaluate_goal_progress(
            goal=goal,
            chat_history=ChatHistory.from_list_of_dicts(self.agent.history),
        )
        self.goal_manager.save_evaluated_goal(
            goal=goal,
            evaluation=goal_evaluation,
        )
        self.agent.history.append(
            ChatMessage(
                role="user",
                content=f"# Goal evaluation\n{goal_evaluation}",
            ).model_dump()
        )
        self.save_agent_history(
            initial_formatted_system_prompt=initial_formatted_system_prompt,
            # Save only the new (last) message, which is the goal evaluation.
            save_last_n=1,
        )


class DeployableMicrochainAgent(DeployableMicrochainAgentAbstract):
    identifier = MICROCHAIN_AGENT_OMEN
    functions_config = TRADING_AGENT_SYSTEM_PROMPT_CONFIG.functions_config
    initial_system_prompt = TRADING_AGENT_SYSTEM_PROMPT_CONFIG.system_prompt


class DeployableMicrochainModifiableSystemPromptAgentAbstract(
    DeployableMicrochainAgent
):
    functions_config = JUST_BORN_SYSTEM_PROMPT_CONFIG.functions_config
    initial_system_prompt = JUST_BORN_SYSTEM_PROMPT_CONFIG.system_prompt


class DeployableMicrochainModifiableSystemPromptAgent0(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    identifier = MICROCHAIN_AGENT_OMEN_LEARNING_0

    @classmethod
    def get_description(cls) -> str:
        return "Microchain agent with 'just born' system prompt, and ability to adjust its own system prompt, version 0."


class DeployableMicrochainModifiableSystemPromptAgent1(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    identifier = MICROCHAIN_AGENT_OMEN_LEARNING_1

    @classmethod
    def get_description(cls) -> str:
        return "Microchain agent with 'just born' system prompt, and ability to adjust its own system prompt, version 1."


class DeployableMicrochainModifiableSystemPromptAgent2(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    identifier = MICROCHAIN_AGENT_OMEN_LEARNING_2

    @classmethod
    def get_description(cls) -> str:
        return "Microchain agent with 'just born' system prompt, and ability to adjust its own system prompt, version 2."


class DeployableMicrochainModifiableSystemPromptAgent3(
    DeployableMicrochainModifiableSystemPromptAgentAbstract
):
    identifier = MICROCHAIN_AGENT_OMEN_LEARNING_3
    model = SupportedModel.llama_31_instruct
    # Force less iterations, because Replicate's API allows at max 4096 input tokens.
    max_iterations = 10

    @classmethod
    def get_description(cls) -> str:
        return "Microchain agent with 'just born' system prompt, and ability to adjust its own system prompt, version 3. Uses Llama 3.1 model."


class DeployableMicrochainWithGoalManagerAgent0(DeployableMicrochainAgent):
    identifier = MICROCHAIN_AGENT_OMEN_WITH_GOAL_MANAGER
    model = SupportedModel.gpt_4o
    functions_config = TRADING_AGENT_SYSTEM_PROMPT_MINIMAL_CONFIG.functions_config
    initial_system_prompt = TRADING_AGENT_SYSTEM_PROMPT_MINIMAL_CONFIG.system_prompt

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
