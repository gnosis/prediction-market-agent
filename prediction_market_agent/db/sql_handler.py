import typing as t

from prediction_market_agent_tooling.tools.db.db_manager import DBManager
from sqlalchemy import BinaryExpression, ColumnElement
from sqlmodel import SQLModel, asc, desc

SQLModelType = t.TypeVar("SQLModelType", bound=SQLModel)


class SQLHandler:
    def __init__(
        self,
        model: t.Type[SQLModelType],
        sqlalchemy_db_url: str | None = None,
    ):
        self.db_manager = DBManager(sqlalchemy_db_url)
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

    def remove_multiple(self, items: t.Sequence[SQLModelType]) -> None:
        with self.db_manager.get_session() as session:
            for item in items:
                session.delete(item)
            session.commit()

    def remove_by_id(self, item_id: int) -> None:
        with self.db_manager.get_session() as session:
            session.query(self.table).filter_by(id=item_id).delete()
            session.commit()

    def get_with_filter_and_order(
        self,
        query_filters: t.Sequence[ColumnElement[bool] | BinaryExpression[bool]] = (),
        order_by_column_name: str | None = None,
        order_desc: bool = True,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[SQLModelType]:
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
            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)
            results = query.all()
        return results

    def count(
        self,
        query_filters: t.Sequence[ColumnElement[bool] | BinaryExpression[bool]] = (),
    ) -> int:
        with self.db_manager.get_session() as session:
            query = session.query(self.table)
            for exp in query_filters:
                query = query.where(exp)
            return query.count()
