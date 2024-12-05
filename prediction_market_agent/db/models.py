from typing import Optional

from prediction_market_agent_tooling.gtypes import (
    HexAddress,
)
from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes
from prediction_market_agent_tooling.tools.utils import DatetimeUTC
from sqlmodel import Field, SQLModel


class LongTermMemories(SQLModel, table=True):
    __tablename__ = "long_term_memories"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    task_description: str
    metadata_: Optional[str] = None
    datetime_: DatetimeUTC


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
    session_identifier: str
    datetime_: DatetimeUTC


class EvaluatedGoalModel(SQLModel, table=True):
    """
    Checkpoint for general agent's goals. Used to store the agent's progress
    towards a goal, and to restore it in future sessions.
    """

    __tablename__ = "evaluated_goals"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    agent_id: str  # Per-agent identifier
    goal: str
    motivation: str
    completion_criteria: str
    is_complete: bool
    reasoning: str
    output: str | None
    datetime_: DatetimeUTC


class BlockchainMessage(SQLModel, table=True):
    """Messages sent to agents via data fields within blockchain transfers."""

    __tablename__ = "blockchain_messages"
    __table_args__ = {
        "extend_existing": True
    }  # required if initializing an existing table
    id: Optional[int] = Field(default=None, primary_key=True)
    consumer_address: HexAddress
    transaction_hash: HexBytes
    block: int
    value_wei: int
