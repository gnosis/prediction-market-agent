import dotenv
import os
import typing as t
from prediction_market_agent.tools.types import HexAddress, PrivateKey


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


def get_bet_from_address() -> HexAddress:
    return HexAddress(get_api_key("BET_FROM_ADDRESS"))


def get_bet_from_private_key() -> PrivateKey:
    return PrivateKey(get_api_key("BET_FROM_PRIVATE_KEY"))


class APIKeys:
    def __init__(
        self,
        manifold: t.Optional[str] = None,
        serp: t.Optional[str] = None,
        openai: t.Optional[str] = None,
        bet_from_address: t.Optional[HexAddress] = None,
        bet_from_private_key: t.Optional[PrivateKey] = None,
    ):
        self.manifold = manifold
        self.serp = serp
        self.openai = openai
        self.bet_from_address = bet_from_address
        self.bet_from_private_key = bet_from_private_key


def get_keys() -> APIKeys:
    return APIKeys(
        manifold=get_manifold_api_key(),
        serp=get_serp_api_key(),
        openai=get_openai_api_key(),
        bet_from_address=get_bet_from_address(),
        bet_from_private_key=get_bet_from_private_key(),
    )


def get_market_prompt(question: str) -> str:
    prompt = (
        f"Research and report on the following question:\n\n"
        f"{question}\n\n"
        f"Return ONLY a single world answer: 'Yes' or 'No', even if you are unsure. If you are unsure, make your best guess.\n"
    )
    return prompt


def parse_result_to_boolean(result: str) -> bool:
    return True if result == "Yes" else False


def parse_result_to_str(result: bool) -> str:
    return "Yes" if result else "No"
