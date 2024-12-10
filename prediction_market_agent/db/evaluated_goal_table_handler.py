import typing as t

from sqlmodel import col

from prediction_market_agent.db.models import EvaluatedGoalModel
from prediction_market_agent.db.sql_handler import SQLHandler


class EvaluatedGoalTableHandler:
    def __init__(
        self,
        agent_id: str,
        sqlalchemy_db_url: str | None = None,
    ):
        self.agent_id = agent_id
        self.sql_handler = SQLHandler(
            model=EvaluatedGoalModel,
            sqlalchemy_db_url=sqlalchemy_db_url,
        )

    def save_evaluated_goal(self, model: EvaluatedGoalModel) -> None:
        self.sql_handler.save_multiple([model])

    def get_latest_evaluated_goals(self, limit: int) -> list[EvaluatedGoalModel]:
        column_to_order: str = EvaluatedGoalModel.datetime_.key  # type: ignore
        items: t.Sequence[
            EvaluatedGoalModel
        ] = self.sql_handler.get_with_filter_and_order(
            query_filters=[col(EvaluatedGoalModel.agent_id) == self.agent_id],
            order_by_column_name=column_to_order,
            order_desc=True,
            limit=limit,
        )
        return list(items)
