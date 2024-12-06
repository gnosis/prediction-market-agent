import typing as t

from prediction_market_agent_tooling.config import APIKeys as APIKeys_PMAT
from prediction_market_agent_tooling.tools.db.db_manager import DBManager
from prediction_market_agent_tooling.tools.utils import check_not_none
from pydantic import SecretStr
from sqlalchemy import BinaryExpression, ColumnElement
from sqlmodel import SQLModel, asc, desc

from prediction_market_agent.utils import DBKeys

SQLModelType = t.TypeVar("SQLModelType", bound=SQLModel)


class SQLHandler:
    def __init__(
        self,
        model: t.Type[SQLModelType],
        sqlalchemy_db_url: str | None = None,
    ):
        self.sqlalchemy_db_url = (
            sqlalchemy_db_url
            if sqlalchemy_db_url is not None
            else check_not_none(DBKeys().SQLALCHEMY_DB_URL).get_secret_value()
        )

        api_keys = APIKeys_PMAT(SQLALCHEMY_DB_URL=SecretStr(self.sqlalchemy_db_url))
        self.db_manager = DBManager(api_keys)
        self.table = model
        self._init_table_if_not_exists()

    def _init_table_if_not_exists(self) -> None:
        self.db_manager.create_tables(sqlmodel_tables=[self.table])

    def get_all(self) -> t.Sequence[SQLModelType]:
        with self.db_manager.get_session() as session:
            return session.query(self.table).all()

    def save_multiple(self, items: t.Sequence[SQLModelType]) -> None:
        with self.db_manager.get_session() as session:
            session.add_all(items)
            session.commit()

    def delete_all_entries(
        self, col_name: str | None = None, col_value: str | None = None
    ) -> None:
        with self.db_manager.get_session() as session:
            query = session.query(self.table)
            if col_name and col_value:
                query = query.filter_by(**{col_name: col_value})
            query.delete()
            session.commit()

    def get_with_filter_and_order(
        self,
        query_filters: t.Sequence[ColumnElement[bool] | BinaryExpression[bool]] = (),
        order_by_column_name: str | None = None,
        order_desc: bool = True,
        limit: int | None = None,
    ) -> t.Sequence[SQLModelType]:
        with self.db_manager.get_session() as session:
            query = session.query(self.table)
            for exp in query_filters:
                query = query.where(exp)

            if order_by_column_name:
                query = query.order_by(
                    desc(order_by_column_name)
                    if order_desc
                    else asc(order_by_column_name)
                )
            if limit:
                query = query.limit(limit)
            results = query.all()
        return results
