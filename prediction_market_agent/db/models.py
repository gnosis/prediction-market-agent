import json
from functools import cached_property
from typing import Any, Optional

from prediction_market_agent_tooling.gtypes import xDaiWei
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.utils import DatetimeUTC
from sqlalchemy import Column, Numeric
from sqlmodel import Field, SQLModel

from prediction_market_agent.agents.microchain_agent.nft_treasury_game.game_history import (
    NFTGameRound,
)


class LongTermMemories(SQLModel, table=True):
    __tablename__ = "long_term_memories"
    __table_args__ = {"extend_existing": True}
    id: Optional[int] = Field(default=None, primary_key=True)
    task_description: str
    metadata_: Optional[str] = None
    datetime_: DatetimeUTC

    @cached_property
    def metadata_dict(self) -> dict[str, Any] | None:
        try:
            out: dict[str, Any] | None = (
                json.loads(self.metadata_) if self.metadata_ else None
            )
            return out
        except Exception as e:
            logger.error(
                f"Error while loading {self.__class__.__name__} with {self.id=} metadata: {self.metadata_} "
            )
            raise e


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
        "extend_existing": True,  # required if initializing an existing table
    }
    id: Optional[int] = Field(default=None, primary_key=True)
    consumer_address: str
    sender_address: str
    transaction_hash: str = Field(unique=True)
    block: str = Field(sa_column=Column(Numeric, nullable=False))
    value_wei: str = Field(sa_column=Column(Numeric, nullable=False))
    data_field: Optional[str]

    @property
    def block_parsed(self) -> int:
        return int(self.block)

    @property
    def value_wei_parsed(self) -> xDaiWei:
        return xDaiWei(self.value_wei)

    def __str__(self) -> str:
        return f"""Sender: {self.sender_address}
Value: {self.value_wei_parsed.as_xdai} xDai
Message: {self.data_field}
"""


class ReportNFTGame(SQLModel, table=True):
    """Reports summarizing activities that took place during the NFT game."""

    __tablename__ = "report_nft_game"
    __table_args__ = {
        "extend_existing": True,
    }
    id: Optional[int] = Field(default=None, primary_key=True)
    game_round_id: int = Field(foreign_key=f"{NFTGameRound.__tablename__}.id")
    agent_id: Optional[str] = (
        None  # we keep it optional to allow for the final summary (involving all agents) to be stored in this table
    )
    # as well.
    learnings: str
    datetime_: DatetimeUTC

    @property
    def is_overall_report(self) -> bool:
        return self.agent_id is None
