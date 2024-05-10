from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class LongTermMemories(SQLModel, table=True):
    __tablename__ = "long_term_memories"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    task_description: str
    metadata_: Optional[str] = None
    datetime_: datetime
