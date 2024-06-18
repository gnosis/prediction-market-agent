from loguru import logger
from prediction_market_agent_tooling.tools.utils import check_not_none
from sqlalchemy import Table
from sqlmodel import create_engine, SQLModel, Session
from sqlmodel.main import SQLModelMetaclass

# from prediction_market_agent.db.models import LongTermMemories, Prompt
import typing as t
from prediction_market_agent.utils import DBKeys

TableType = t.TypeVar("TableType", bound=SQLModel)


class SqlHandler(t.Generic[TableType]):
    def __init__(self, model: t.Type[TableType], sqlalchemy_db_url: str | None = None):
        self.engine = create_engine(
            sqlalchemy_db_url
            if sqlalchemy_db_url
            else check_not_none(DBKeys().SQLALCHEMY_DB_URL)
        )
        self.table = model
        self._init_table_if_not_exists()

    def _init_table_if_not_exists(self) -> None:
        table = SQLModel.metadata.tables[str(self.table.__tablename__)]
        SQLModel.metadata.create_all(self.engine, tables=[table])

    def select_all(self) -> list[TableType]:
        return Session(self.engine).query(self.table).all()

    def save_multiple(self, items: list[TableType]) -> None:
        with Session(self.engine) as session:
            session.add_all(items)
            session.commit()
