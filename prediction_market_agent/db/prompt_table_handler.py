from prediction_market_agent_tooling.tools.utils import utcnow
from sqlmodel import col
import typing as t
from prediction_market_agent.db.models import PROMPT_DEFAULT_SESSION_IDENTIFIER, Prompt
from prediction_market_agent.db.sql_handler import SQLHandler


class PromptTableHandler:
    def __init__(
        self,
        session_identifier: str | None = None,
        sqlalchemy_db_url: str | None = None,
    ):
        self.session_identifier = session_identifier
        self.sql_handler = SQLHandler(model=Prompt, sqlalchemy_db_url=sqlalchemy_db_url)

    def save_prompt(self, prompt: str) -> None:
        """Save item to storage."""
        prompt_to_save = Prompt(
            prompt=prompt,
            datetime_=utcnow(),
            session_identifier=self.session_identifier
            if self.session_identifier
            else PROMPT_DEFAULT_SESSION_IDENTIFIER,
        )
        self.sql_handler.save_multiple([prompt_to_save])

    def fetch_latest_prompt(
        self, session_identifier: str = PROMPT_DEFAULT_SESSION_IDENTIFIER
    ) -> Prompt | None:
        # We ignore since mypy doesn't play well with SQLModel class attributes.
        column_to_order: str = Prompt.datetime_.key  # type: ignore
        items: t.Sequence[Prompt] = self.sql_handler.get_with_filter_and_order(
            query_filters=[col(Prompt.session_identifier) == session_identifier],
            order_by_column_name=column_to_order,
            order_desc=True,
            limit=1,
        )
        return items[0] if items else None

    def fetch_latest_prompt123(self) -> Prompt | None:
        aaa
        return self.storage.load_latest_prompt(
            session_identifier=self.session_identifier
            if self.session_identifier
            else PROMPT_DEFAULT_SESSION_IDENTIFIER
        )
