from pydantic import BaseModel

from prediction_market_agent.agents.microchain_agent.memory import ChatHistory


class Goal(BaseModel):
    prompt: str
    motivation: str
    completion_criteria: str  # ?


class EvaluatedGoal(Goal):
    is_complete: bool
    reasoning: str
    output: str | None  # or 'learning'?

    def to_goal(self) -> Goal:
        return Goal(
            prompt=self.prompt,
            motivation=self.motivation,
            completion_criteria=self.completion_criteria,
        )


class GoalManager:
    def __init__(
        self,
        agent_id: str,
    ):
        self.agent_id: str = agent_id

    def get_latest_goal_from_memory(self) -> EvaluatedGoal | None:
        pass

    def generate_goal(self) -> Goal:
        """
        If a goal exists from a previous session, load it and check its status.
        Otherwise create a new one.
        """
        pass

    def get_goal(self) -> Goal:
        if goal := self.get_latest_goal_from_memory():
            if goal.is_complete:
                # Generate a new goal
                return self.generate_goal()
            else:
                # Try again
                return goal
        return self.generate_goal()

    def evaluate_goal_progress(
        goal: Goal,
        chat_history: ChatHistory,
    ) -> EvaluatedGoal:
        pass

    def save_evaluated_goal(self, goal: EvaluatedGoal) -> None:
        pass
