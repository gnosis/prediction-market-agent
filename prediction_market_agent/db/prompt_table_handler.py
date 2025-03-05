import typing as t

from prediction_market_agent_tooling.tools.utils import utcnow
from sqlmodel import col

from prediction_market_agent.agents.identifiers import AgentIdentifier
from prediction_market_agent.db.models import Prompt
from prediction_market_agent.db.sql_handler import SQLHandler


class PromptTableHandler:
    def __init__(
        self,
        session_identifier: str,
        sqlalchemy_db_url: str | None = None,
    ):
        self.session_identifier = session_identifier
        self.sql_handler = SQLHandler(model=Prompt, sqlalchemy_db_url=sqlalchemy_db_url)

    @staticmethod
    def from_agent_identifier(
        identifier: AgentIdentifier,
    ) -> "PromptTableHandler":
        return PromptTableHandler(session_identifier=identifier)

    def save_prompt(self, prompt: str) -> None:
        """Save item to storage."""
        prompt_to_save = Prompt(
            prompt=prompt,
            datetime_=utcnow(),
            session_identifier=self.session_identifier,
        )
        self.sql_handler.save_multiple([prompt_to_save])

    def fetch_latest_prompt(self) -> Prompt | None:
        # We ignore since mypy doesn't play well with SQLModel class attributes.
        column_to_order: str = Prompt.datetime_.key  # type: ignore[attr-defined]
        query_filters = [col(Prompt.session_identifier) == self.session_identifier]
        items: t.Sequence[Prompt] = self.sql_handler.get_with_filter_and_order(
            query_filters=query_filters,
            order_by_column_name=column_to_order,
            order_desc=True,
            limit=1,
        )

        return items[0] if items else None
