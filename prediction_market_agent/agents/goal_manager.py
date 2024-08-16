from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic import BaseModel

from prediction_market_agent.agents.microchain_agent.memory import ChatHistory
from prediction_market_agent.db.evaluated_goal_table_handler import (
    EvaluatedGoalTableHandler,
)
from prediction_market_agent.db.models import EvaluatedGoalModel


class Goal(BaseModel):
    prompt: str
    motivation: str
    completion_criteria: str  # TODO maybe?


class EvaluatedGoal(Goal):
    is_complete: bool
    reasoning: str
    output: str | None

    @classmethod
    def from_model(cls, model: EvaluatedGoalModel) -> "EvaluatedGoal":
        return EvaluatedGoal(
            prompt=model.prompt,
            motivation=model.motivation,
            completion_criteria=model.completion_criteria,
            is_complete=model.is_complete,
            reasoning=model.reasoning,
            output=model.output,
        )

    def to_model(self, agent_id: str) -> EvaluatedGoalModel:
        return EvaluatedGoalModel(
            prompt=self.prompt,
            motivation=self.motivation,
            completion_criteria=self.completion_criteria,
            is_complete=self.is_complete,
            reasoning=self.reasoning,
            output=self.output,
            agent_id=agent_id,
            datetime_=utcnow(),
        )


class GoalManager:
    def __init__(
        self,
        agent_id: str,
        sqlalchemy_db_url: str | None = None,
    ):
        self.agent_id = agent_id
        self.table_handler = EvaluatedGoalTableHandler(
            agent_id=agent_id,
            sqlalchemy_db_url=sqlalchemy_db_url,
        )

    def get_latest_evaluated_goal_from_memory(self) -> EvaluatedGoal | None:
        evaluated_goal_model = self.table_handler.get_latest_evaluated_goal()
        if evaluated_goal_model:
            return EvaluatedGoal.from_model(model=evaluated_goal_model)
        return None

    def generate_goal(self) -> Goal:
        """
        If a goal exists from a previous session, load it and check its status.
        Otherwise create a new one.
        """
        # TODO
        return Goal(
            prompt="foo",
            motivation="bar",
            completion_criteria="baz",
        )

    def get_goal(self) -> Goal:
        if goal := self.get_latest_evaluated_goal_from_memory():
            if goal.is_complete:
                # Generate a new goal
                return self.generate_goal()
            else:
                # Try again
                return goal
        return self.generate_goal()

    def evaluate_goal_progress(
        self,
        goal: Goal,
        chat_history: ChatHistory,
    ) -> EvaluatedGoal:
        # TODO
        return EvaluatedGoal(
            prompt=goal.prompt,
            motivation=goal.motivation,
            completion_criteria=goal.completion_criteria,
            is_complete=False,
            reasoning="",
            output="",
        )

    def save_evaluated_goal(self, goal: EvaluatedGoal) -> None:
        model = goal.to_model(agent_id=self.agent_id)
        self.table_handler.save_evaluated_goal(model)
