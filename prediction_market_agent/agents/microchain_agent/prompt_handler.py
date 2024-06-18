from loguru import logger
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.db.db_storage import DBStorage
from prediction_market_agent.db.models import Prompt, PROMPT_DEFAULT_SESSION_IDENTIFIER


# ToDo - Unify PromptHandler, db_storage and LongTermMemory into 2 classes, one per table.
class PromptHandler:
    def __init__(
        self,
        session_identifier: str | None = None,
        sqlalchemy_db_url: str | None = None,
    ):
        self.session_identifier = session_identifier
        self.storage = DBStorage(sqlalchemy_db_url=sqlalchemy_db_url)
        self.storage._initialize_db()

    def save_prompt(self, prompt: str) -> None:
        """Save item to storage."""
        prompt_to_save = Prompt(
            prompt=prompt,
            datetime_=utcnow(),
            session_identifier=self.session_identifier
            if self.session_identifier
            else PROMPT_DEFAULT_SESSION_IDENTIFIER,
        )
        self.storage.save_multiple([prompt_to_save])

    def fetch_latest_prompt(
        self, session_identifier: str | None = None
    ) -> Prompt | None:
        return self.storage.load_latest_prompt(session_identifier=session_identifier)
