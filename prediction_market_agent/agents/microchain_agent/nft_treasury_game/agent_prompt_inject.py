from sqlmodel import Field, SQLModel, String, col

from prediction_market_agent.agents.identifiers import AgentIdentifier
from prediction_market_agent.db.sql_handler import SQLHandler


class AgentPromptInject(SQLModel, table=True):
    """
    TODO: Should this be joined with AgentRegistry?
    """

    __table_args__ = {
        "extend_existing": True
    }  # required if initializing an existing table
    id: int | None = Field(default=None, primary_key=True)
    agent_identifier: AgentIdentifier = Field(sa_type=String)
    prompt: str


class PromptInjectHandler:
    def __init__(
        self,
        agent_identifier: AgentIdentifier,
        sqlalchemy_db_url: str | None = None,
    ):
        self.agent_identifier = agent_identifier
        self.sql_handler = SQLHandler(
            model=AgentPromptInject, sqlalchemy_db_url=sqlalchemy_db_url
        )

    @staticmethod
    def from_agent_identifier(
        identifier: AgentIdentifier,
    ) -> "PromptInjectHandler":
        return PromptInjectHandler(agent_identifier=identifier)

    def get(self) -> AgentPromptInject | None:
        prompts: list[AgentPromptInject] = self.sql_handler.get_with_filter_and_order(
            query_filters=[
                col(AgentPromptInject.agent_identifier) == self.agent_identifier
            ],
            limit=1,
        )
        return prompts[0] if prompts else None

    def add(self, prompt: str) -> None:
        item = AgentPromptInject(agent_identifier=self.agent_identifier, prompt=prompt)
        self.sql_handler.save_multiple([item])
