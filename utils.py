import dotenv
import os


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


def get_market_prompt(question: str) -> str:
    prompt = """
    Research and report on the following question:

    {}

    Return a single world answer: 'Yes' or 'No'. If you are unsure, make your best guess.
    """
    return prompt.format(question)


def parse_result_to_boolean(result: str) -> bool:
    return True if result == "Yes" else False
