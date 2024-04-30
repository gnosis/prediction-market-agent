import json
from typing import Union, Dict, Any

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from prediction_market_agent.utils import DBKeys
from datetime import datetime


class TableOutput(BaseModel):
    metadata: dict[str, str]
    datetime: datetime
    score: float


class DBStorage:
    def __init__(self):
        keys = DBKeys()
        if not keys.sqlalchemy_db_url:
            raise EnvironmentError(
                "Cannot initialize DBHandler without a valid sqlalchemy_db_url"
            )
        self.engine = create_engine(keys.sqlalchemy_db_url)

    def _initialize_db(self):
        """
        Initializes the SQLite database and creates LTM table
        """
        try:
            with self.engine.connect() as con:
                con.execute(
                    text(
                        """CREATE TABLE IF NOT EXISTS long_term_memories (
                        id SERIAL primary KEY,
                        task_description TEXT not NULL,
                        metadata TEXT,
                        datetime TIMESTAMPTZ DEFAULT NOW(),
                        score real                        
                    );
                """
                    )
                )
                con.commit()
        except Exception as e:
            logger.warning("Could not instantiate table ", e)

    def save(
        self,
        task_description: str,
        metadata: Dict[str, Any],
        score: Union[int, float],
    ) -> None:
        """Saves data to the LTM table with error handling."""
        try:
            params = dict(x1=task_description, x2=json.dumps(metadata), x3=score)
            with self.engine.connect() as con:
                variables = ",".join([f":{i}" for i in params.keys()])  #':x1,:x2'
                values_as_tuple = f"({variables})"  # (:x1,:x2)
                stmt = text(
                    f"""
                INSERT INTO long_term_memories (task_description, metadata, score)
                VALUES {values_as_tuple}"""
                )
                # stmt.bindparams(**params)
                con.execute(stmt, params)
                con.commit()
        except Exception as e:
            logger.error(
                f"MEMORY ERROR: An error occurred while saving to LTM: {e}",
            )

    def load(self, task_description: str, latest_n: int) -> list[TableOutput]:
        """Queries the LTM table by task description with error handling."""
        key = "task_description"
        try:
            with self.engine.connect() as con:
                rows = con.execute(
                    text(
                        f"""
                    SELECT metadata, datetime, score
                    FROM long_term_memories
                    WHERE task_description = :{key}
                    ORDER BY datetime DESC, score ASC
                    LIMIT {latest_n if latest_n else 1000}
                """
                    ),
                    {key: task_description},
                ).fetchall()
                if rows:
                    return [
                        TableOutput.model_validate(
                            {
                                "metadata": json.loads(row[0]),
                                "datetime": row[1],
                                "score": row[2],
                            }
                        )
                        for row in rows
                    ]

        except Exception as e:
            logger.error(f"MEMORY ERROR: An error occurred while querying LTM: {e}")
        return []
