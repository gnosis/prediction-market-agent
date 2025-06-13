import os
import subprocess
import urllib.parse
from pathlib import Path

import typer
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.tools.utils import check_not_none, utcnow


def main(table: list[str]) -> None:
    """
    You need to have pg_dump installed to run this. On Mac: `brew install postgres@15`.

    Run as
    ```
    python prediction_market_agent/agents/microchain_agent/sql/export_data.py agentdb long_term_memories nft_game_round report_nft_game prompts
    ```
    """
    api_keys = APIKeys()

    dumps_folder = Path(__file__).parent / "dumps"
    dumps_folder.mkdir(exist_ok=True, parents=True)

    for table_ in table:
        db_url = api_keys.sqlalchemy_db_url.get_secret_value()
        parsed = urllib.parse.urlparse(db_url)

        dbname = parsed.path.lstrip("/")
        user = check_not_none(parsed.username)
        password = check_not_none(parsed.password)
        host = check_not_none(parsed.hostname)
        port = str(parsed.port) if parsed.port else "5432"

        dump_path = f"{dumps_folder}/{table_}.{utcnow()}.dump"
        cmd = [
            "pg_dump",
            "-h",
            host,
            "-p",
            port,
            "-U",
            user,
            "-d",
            dbname,
            "-t",
            table_,
            "-F",
            "p",
            "-f",
            dump_path,
        ]
        subprocess.run(cmd, env=os.environ | {"PGPASSWORD": password})


if __name__ == "__main__":
    typer.run(main)
