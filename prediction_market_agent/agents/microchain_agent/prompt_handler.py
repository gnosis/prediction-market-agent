from loguru import logger
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.db.db_storage import DBStorage
from prediction_market_agent.db.models import Prompt


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
        """Save item to storage. Note that score allows many types for easier handling by agent."""
        if not prompt:
            logger.info("No prompt to save")
            return
        prompt = Prompt(
            prompt=prompt,
            datetime_=utcnow(),
            session_identifier=self.session_identifier,
        )
        self.storage.save_multiple([prompt])

    def fetch_latest_prompt(
        self, session_identifier: str | None = None
    ) -> Prompt | None:
        return self.storage.load_latest_prompt(session_identifier=session_identifier)
