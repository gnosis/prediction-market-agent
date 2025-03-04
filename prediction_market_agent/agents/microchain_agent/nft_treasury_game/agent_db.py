from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import ChecksumAddress, private_key_type
from sqlmodel import Field, SQLModel, String, col

from prediction_market_agent.agents.identifiers import (
    AgentIdentifier,
    build_nft_treasury_game_agent_identifier,
)
from prediction_market_agent.db.sql_handler import SQLHandler


class AgentDB(SQLModel, table=True):
    """
    TODO: Should this be joined with AgentRegistry?
    """

    __table_args__ = {
        "extend_existing": True
    }  # required if initializing an existing table
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    initial_system_prompt: str
    private_key: str
    safe_address: ChecksumAddress | None = Field(sa_type=String, nullable=True)

    @property
    def identifier(self) -> AgentIdentifier:
        return build_nft_treasury_game_agent_identifier(self.name)

    @property
    def api_keys(self) -> APIKeys:
        return APIKeys(
            BET_FROM_PRIVATE_KEY=private_key_type(self.private_key),
            SAFE_ADDRESS=self.safe_address,
        )

    @property
    def wallet_address(self) -> ChecksumAddress:
        return self.api_keys.bet_from_address


class AgentTableHandler:
    def __init__(
        self,
        sqlalchemy_db_url: str | None = None,
    ):
        self.sql_handler = SQLHandler(
            model=AgentDB, sqlalchemy_db_url=sqlalchemy_db_url
        )

    def get(self, agent_identifier: AgentIdentifier) -> AgentDB:
        return self.sql_handler.get_with_filter_and_order(
            query_filters=[col(AgentDB.name) == agent_identifier],
            limit=1,
        )[0]

    def add_agent(self, agent: AgentDB) -> None:
        self.sql_handler.save_multiple([agent])
