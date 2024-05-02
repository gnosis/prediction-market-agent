from datetime import datetime
from typing import Optional

from prediction_market_agent_tooling.tools.utils import utcnow
from sqlmodel import SQLModel, Field


class LongTermMemories(SQLModel, table=True):
    __tablename__ = "long_term_memories"
    id: Optional[int] = Field(default=None, primary_key=True)
    task_description: str
    metadata_: Optional[str] = None
    datetime_: datetime
