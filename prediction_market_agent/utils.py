import typing as t
from prediction_market_agent_tooling.tools.utils import (
    check_not_none,
    should_not_happen,
)
from prediction_market_agent_tooling.config import APIKeys as APIKeysBase


class APIKeys(APIKeysBase):
    SERP_API_KEY: t.Optional[str] = None
    OPENAI_API_KEY: t.Optional[str] = None

    @property
    def serp_api_key(self) -> str:
        return check_not_none(  # type: ignore  # Remove once PMAT is correctly released and this doesn't ignore his typing.
            self.SERP_API_KEY, "SERP_API_KEY missing in the environment."
        )

    @property
    def openai_api_key(self) -> str:
        return check_not_none(  # type: ignore  # Remove once PMAT is correctly released and this doesn't ignore his typing.
            self.OPENAI_API_KEY, "OPENAI_API_KEY missing in the environment."
        )


def get_market_prompt(question: str) -> str:
    prompt = (
        f"Research and report on the following question:\n\n"
        f"{question}\n\n"
        f"Return ONLY a single world answer: 'Yes' or 'No', even if you are unsure. If you are unsure, make your best guess.\n"
    )
    return prompt


def parse_result_to_boolean(result: str) -> bool:
    return (  # type: ignore  # Remove once PMAT is correctly released and this doesn't ignore his typing.
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
