from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.tools.langfuse_ import (
    get_langfuse_langchain_config,
    observe,
)
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic import BaseModel, Field

from prediction_market_agent.agents.microchain_agent.memory import ChatHistory
from prediction_market_agent.db.evaluated_goal_table_handler import (
    EvaluatedGoalTableHandler,
)
from prediction_market_agent.db.models import EvaluatedGoalModel
from prediction_market_agent.utils import DEFAULT_OPENAI_MODEL, APIKeys

GENERATE_GOAL_PROMPT_TEMPLATE = """
Generate a specific goal for an open-ended, autonomous agent that has a high-level description and a number of specific capabilities.

The goal should satisfy the following:
- have a narrow focus
- be completable immediately, within a single session
- be realistically achievable given the agen't specific capabilities
- have a clear motivation and completion criteria
- advance the aims of the agent
- balance the need for exploration and exploitation
- not be contingent on external factors that are out of the agent's control

If applicable, use the agent's previous evaluated goals when considering its new goal, and state how this goal follows from the previous ones in the 'reasoning' field.

[HIGH LEVEL DESCRIPTION]
{high_level_description}

[AGENT CAPABILITIES]
{agent_capabilities}

[PREVIOUS EVALUATED GOALS]
{previous_evaluated_goals}

{format_instructions}
"""

EVALUATE_GOAL_PROGRESS_PROMPT_TEMPLATE = """
An agent and user are working together to achieve a well defined goal.
Given their chat history, and the goal definition, evaluate whether the goal has been completed.

[GOAL]
{goal_prompt}

[CHAT HISTORY]
{chat_history}

{format_instructions}
"""


class Goal(BaseModel):
    goal: str = Field(..., description="A clear description of the goal")
    motivation: str = Field(..., description="The reason for the goal")
    completion_criteria: str = Field(
        ...,
        description="The criteria that will be used to evaluate whether the goal has been completed",
    )

    def to_prompt(self) -> str:
        return (
            f"# Goal:\n"
            f"{self.goal}\n\n"
            f"## Motivation:\n{self.motivation}\n\n"
            f"## Completion Criteria:\n{self.completion_criteria}"
        )


class GoalEvaluation(BaseModel):
    reasoning: str = Field(
        ..., description="An explanation of why the goal is deemed completed or not"
    )
    is_complete: bool = Field(..., description="Whether the goal is complete")
    output: str | None = Field(
        ...,
        description="If the goal description implied a 'return value', and the goal is complete, this field should contain the output",
    )

    def __str__(self) -> str:
        return (
            f"Is Complete: {self.is_complete}\n"
            f"Reasoning: {self.reasoning}\n"
            f"Output: {self.output}"
        )


class EvaluatedGoal(Goal):
    reasoning: str
    is_complete: bool
    output: str | None

    def __str__(self) -> str:
        return (
            f"Goal: {self.goal}\n"
            f"Motivation: {self.motivation}\n"
            f"Completion Criteria: {self.completion_criteria}\n"
            f"Is Complete: {self.is_complete}\n"
            f"Reasoning: {self.reasoning}\n"
            f"Output: {self.output}"
        )

    @classmethod
    def from_model(cls, model: EvaluatedGoalModel) -> "EvaluatedGoal":
        return EvaluatedGoal(
            goal=model.goal,
            motivation=model.motivation,
            completion_criteria=model.completion_criteria,
            is_complete=model.is_complete,
            reasoning=model.reasoning,
            output=model.output,
        )

    def to_model(self, agent_id: str) -> EvaluatedGoalModel:
        return EvaluatedGoalModel(
            goal=self.goal,
            motivation=self.motivation,
            completion_criteria=self.completion_criteria,
            is_complete=self.is_complete,
            reasoning=self.reasoning,
            output=self.output,
            agent_id=agent_id,
            datetime_=utcnow(),
        )

    def to_goal(self) -> Goal:
        return Goal(
            goal=self.goal,
            motivation=self.motivation,
            completion_criteria=self.completion_criteria,
        )


class GoalManager:
    def __init__(
        self,
        agent_id: str,
        high_level_description: str,
        agent_capabilities: str,
        retry_limit: int = 3,
        model: str = DEFAULT_OPENAI_MODEL,
        sqlalchemy_db_url: str | None = None,
    ):
        self.agent_id = agent_id
        self.high_level_description = high_level_description
        self.agent_capabilities = agent_capabilities
        self.retry_limit = retry_limit
        self.model = model
        self.table_handler = EvaluatedGoalTableHandler(
            agent_id=agent_id,
            sqlalchemy_db_url=sqlalchemy_db_url,
        )

    def get_latest_evaluated_goals_from_memory(self, limit: int) -> list[EvaluatedGoal]:
        evaluated_goal_models = self.table_handler.get_latest_evaluated_goals(
            limit=limit
        )
        return [EvaluatedGoal.from_model(model) for model in evaluated_goal_models]

    @observe()
    def generate_goal(self, latest_evaluated_goals: list[EvaluatedGoal]) -> Goal:
        """
        Generate a new goal based on the high-level description and the latest
        evaluated goals.

        TODO support generation of long-horizon goals with a specified
        completion date, until which the goal's status is 'pending'.
        """
        parser = PydanticOutputParser(pydantic_object=Goal)
        prompt = PromptTemplate(
            template=GENERATE_GOAL_PROMPT_TEMPLATE,
            input_variables=[
                "high_level_description",
                "agent_capabilities",
                "previous_evaluated_goals",
            ],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        latest_evaluated_goals_str = self.evaluated_goals_to_str(latest_evaluated_goals)
        llm = ChatOpenAI(
            temperature=0,
            model=self.model,
            api_key=APIKeys().openai_api_key_secretstr_v1,
        )
        chain = prompt | llm | parser

        goal: Goal = chain.invoke(
            {
                "high_level_description": self.high_level_description,
                "agent_capabilities": self.agent_capabilities,
                "previous_evaluated_goals": latest_evaluated_goals_str,
            },
            config=get_langfuse_langchain_config(),
        )
        return goal

    def have_reached_retry_limit(
        self, latest_evaluated_goals: list[EvaluatedGoal]
    ) -> bool:
        if self.retry_limit == 0:
            return True

        if len(latest_evaluated_goals) < self.retry_limit + 1:
            return False

        latest_goal = latest_evaluated_goals[0].to_goal()
        return all(
            g.to_goal() == latest_goal
            for g in latest_evaluated_goals[: self.retry_limit + 1]
        )

    @observe()
    def get_goal(self) -> Goal:
        """
        Manage the fetching of goals from memory, and deciding when to generate
        a new goal vs. retrying an incomplete one.

        TODO add the ability to continue from a previous session if the goal
        is not complete.
        """
        latest_evaluated_goals = self.get_latest_evaluated_goals_from_memory(
            limit=self.retry_limit
        )
        if latest_evaluated_goals:
            # Previous goals have been retrieved from memory. Generate a new
            # goal based on these, or retry the last on if it did not complete.
            latest_evaluated_goal = latest_evaluated_goals[0]

            if latest_evaluated_goal.is_complete:
                # Generate a new goal
                return self.generate_goal(latest_evaluated_goals)
            else:
                # Try again, unless we've reached the retry limit
                if self.have_reached_retry_limit(latest_evaluated_goals):
                    return self.generate_goal(latest_evaluated_goals)
                else:
                    return latest_evaluated_goal.to_goal()

        # No evaluated goals in memory. Generate a new goal from scratch
        return self.generate_goal(latest_evaluated_goals=[])

    @classmethod
    def get_chat_history_after_goal_prompt(
        cls, goal: Goal, chat_history: ChatHistory
    ) -> ChatHistory:
        """
        Return the chat history after the goal prompt, or None if the goal
        prompt is not found.
        """
        for i, chat_message in enumerate(chat_history.chat_messages):
            if chat_message.content == goal.to_prompt():
                return ChatHistory(chat_messages=chat_history.chat_messages[i + 1 :])
        raise ValueError("Goal prompt not found in chat history")

    @observe()
    def evaluate_goal_progress(
        self,
        goal: Goal,
        chat_history: ChatHistory,
    ) -> GoalEvaluation:
        relevant_chat_history = self.get_chat_history_after_goal_prompt(
            goal=goal,
            chat_history=chat_history,
        )
        parser = PydanticOutputParser(pydantic_object=GoalEvaluation)
        prompt = PromptTemplate(
            template=EVALUATE_GOAL_PROGRESS_PROMPT_TEMPLATE,
            input_variables=["goal_prompt", "chat_history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm = ChatOpenAI(
            temperature=0,
            model=self.model,
            api_key=APIKeys().openai_api_key_secretstr_v1,
        )
        chain = prompt | llm | parser

        goal_evaluation: GoalEvaluation = chain.invoke(
            {
                "goal_prompt": goal.to_prompt(),
                "chat_history": str(relevant_chat_history),
            },
            config=get_langfuse_langchain_config(),
        )
        return goal_evaluation

    def save_evaluated_goal(self, goal: Goal, evaluation: GoalEvaluation) -> None:
        evaluated_goal = EvaluatedGoal(
            goal=goal.goal,
            motivation=goal.motivation,
            completion_criteria=goal.completion_criteria,
            is_complete=evaluation.is_complete,
            reasoning=evaluation.reasoning,
            output=evaluation.output,
        )
        model = evaluated_goal.to_model(agent_id=self.agent_id)
        self.table_handler.save_evaluated_goal(model)

    @staticmethod
    def evaluated_goals_to_str(evaluated_goals: list[EvaluatedGoal]) -> str:
        if not evaluated_goals:
            return "-- None --"
        goals_str = ""
        for i, goal in enumerate(evaluated_goals):
            goals_str += f"## Goal {i+1}:\n{goal}\n"
            if i < len(evaluated_goals) - 1:
                goals_str += "\n"
        return goals_str
