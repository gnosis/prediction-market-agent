import typing as t

from prediction_market_agent_tooling.tools.utils import check_not_none
from sqlalchemy import BinaryExpression, ColumnElement
from sqlmodel import Session, SQLModel, asc, create_engine, desc

from prediction_market_agent.utils import DBKeys

SQLModelType = t.TypeVar("SQLModelType", bound=SQLModel)


class SQLHandler:
    def __init__(self, model: t.Type[SQLModelType], sqlalchemy_db_url: str | None = None):
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

    def get_all(self) -> t.Sequence[SQLModelType]:
        return Session(self.engine).query(self.table).all()

    def save_multiple(self, items: t.Sequence[SQLModelType]) -> None:
        with Session(self.engine) as session:
            session.add_all(items)
            session.commit()

    def get_with_filter_and_order(
        self,
        query_filters: t.Sequence[ColumnElement[bool] | BinaryExpression[bool]] = (),
        order_by_column_name: str | None = None,
        order_desc: bool = True,
        limit: int | None = None,
    ) -> t.Sequence[SQLModelType]:
        with Session(self.engine) as session:
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
