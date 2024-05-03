import asyncio
import json
import typing as t

from loguru import logger
from prediction_market_agent_tooling.config import APIKeys as APIKeysBase
from prediction_market_agent_tooling.tools.utils import (
    check_not_none,
    should_not_happen,
)
from pydantic import SecretStr


class APIKeys(APIKeysBase):
    SERP_API_KEY: t.Optional[SecretStr] = None
    OPENAI_API_KEY: t.Optional[SecretStr] = None
    TAVILY_API_KEY: t.Optional[SecretStr] = None

    @property
    def serp_api_key(self) -> SecretStr:
        return check_not_none(
            self.SERP_API_KEY, "SERP_API_KEY missing in the environment."
        )

    @property
    def openai_api_key(self) -> SecretStr:
        return check_not_none(
            self.OPENAI_API_KEY, "OPENAI_API_KEY missing in the environment."
        )

    @property
    def tavily_api_key(self) -> SecretStr:
        return check_not_none(
            self.TAVILY_API_KEY, "OPENAI_API_KEY missing in the environment."
        )


def get_market_prompt(question: str) -> str:
    prompt = (
        f"Research and report on the following question:\n\n"
        f"{question}\n\n"
        f"Return ONLY a single world answer: 'Yes' or 'No', even if you are unsure. If you are unsure, make your best guess.\n"
    )
    return prompt


def parse_result_to_boolean(result: str) -> bool:
    return (
        True
        if result.lower() == "yes"
        else (
            False
            if result.lower() == "no"
            else should_not_happen(f"Invalid result: {result}")
        )
    )


def parse_result_to_str(result: bool) -> str:
    return "Yes" if result else "No"


def completion_str_to_json(completion: str) -> dict[str, t.Any]:
    """
    Cleans completion JSON in form of a string:

    ```json
    {
        ...
    }
    ```

    into just { ... }
    ```
    """
    start_index = completion.find("{")
    end_index = completion.rfind("}")
    completion = completion[start_index : end_index + 1]
    completion_dict: dict[str, t.Any] = json.loads(completion)
    return completion_dict


def patch_sqlite3() -> None:
    """
    Helps in the environemnt where one can't update system's sqlite3 installation, for example, Streamlit Cloud, where we get:

    ```
    Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0.
    ```

    This function patches the sqlite3 module to use pysqlite3 instead of sqlite3.
    """
    try:
        __import__("pysqlite3")
        import sys

        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        logger.warning("pysqlite3-binary not found, using sqlite3 instead.")


def streamlit_asyncio_event_loop_hack() -> asyncio.AbstractEventLoop:
    """
    This function is a hack to make Streamlit work with asyncio event loop.
    See https://github.com/streamlit/streamlit/issues/744
    """

    def get_or_create_eventloop() -> asyncio.AbstractEventLoop:
        try:
            return asyncio.get_event_loop()
        except RuntimeError as ex:
            if "There is no current event loop in thread" in str(ex):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return asyncio.get_event_loop()
            else:
                raise ex

    loop = get_or_create_eventloop()
    asyncio.set_event_loop(loop)
    return loop
