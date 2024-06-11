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


PROMPT_DEFAULT_SESSION_IDENTIFIER = "microchain-streamlit"


class Prompt(SQLModel, table=True):
    """Checkpoint for general agent's prompts, as a way to restore its past progress."""

    __tablename__ = "prompts"
    __table_args__ = {
        "extend_existing": True
    }  # required if initializing an existing table
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt: str
    # This allows for future distinction between user sessions, if prompts from a specific
    # user (or app) should be persisted.
    session_identifier: Optional[str] = Field(default=PROMPT_DEFAULT_SESSION_IDENTIFIER)
    datetime_: datetime
