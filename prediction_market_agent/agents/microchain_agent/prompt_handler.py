from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.db.db_storage import DBStorage
from prediction_market_agent.db.models import Prompt


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
