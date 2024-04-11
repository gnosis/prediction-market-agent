import typing as t

from prediction_market_agent_tooling.config import APIKeys as APIKeysBase
from prediction_market_agent_tooling.gtypes import PrivateKey
from prediction_market_agent_tooling.tools.utils import (
    check_not_none,
    should_not_happen,
)
from pydantic import SecretStr


class APIKeys(APIKeysBase):
    SERP_API_KEY: t.Optional[SecretStr] = None
    OPENAI_API_KEY: t.Optional[SecretStr] = None
    TAVILY_API_KEY: t.Optional[SecretStr] = None
    BET_FROM_PRIVATE_KEY: t.Optional[SecretStr] = None

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

    @property
    def bet_from_private_key(self) -> PrivateKey:
        return check_not_none(
            self.BET_FROM_PRIVATE_KEY,
            "BET_FROM_PRIVATE_KEY missing in the environment.",
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
