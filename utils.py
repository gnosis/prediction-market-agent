import dotenv
import os
import typing as t


def get_api_key(name: str) -> str:
    dotenv.load_dotenv()
    key = os.getenv(name)
    if not key:
        raise Exception(f"No API key found. Please set env var '{name}'.")
    return key


def get_manifold_api_key() -> str:
    return get_api_key("MANIFOLD_API_KEY")


def get_serp_api_key() -> str:
    return get_api_key("SERP_API_KEY")


def get_openai_api_key() -> str:
    return get_api_key("OPENAI_API_KEY")


class APIKeys:
    def __init__(
        self,
        manifold: t.Optional[str] = None,
        serp: t.Optional[str] = None,
        openai: t.Optional[str] = None,
    ):
        self.manifold = manifold
        self.serp = serp
        self.openai = openai


def get_keys() -> APIKeys:
    return APIKeys(
        manifold=get_manifold_api_key(),
        serp=get_serp_api_key(),
        openai=get_openai_api_key(),
    )


def get_market_prompt(question: str) -> str:
    prompt = (
        f"Research and report on the following question:\n\n"
        f"{question}\n\n"
        f"Return a single world answer: 'Yes' or 'No'. If you are unsure, make your best guess.\n"
    )
    return prompt


def parse_result_to_boolean(result: str) -> bool:
    return True if result == "Yes" else False
