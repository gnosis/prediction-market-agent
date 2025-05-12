from langfuse.openai import AsyncOpenAI
from openai import DEFAULT_TIMEOUT, DefaultAsyncHttpxClient
from pydantic import SecretStr
from pydantic_ai.models.openai import OpenAIModel  # noqa: F401 # Just for convenience.
from pydantic_ai.providers.openai import OpenAIProvider

OPENAI_BASE_URL = "https://api.openai.com/v1"


def get_openai_provider(
    api_key: SecretStr,
    base_url: str = OPENAI_BASE_URL,
) -> OpenAIProvider:
    """
    For some reason, when OpenAIProvider/AsyncOpenAI is initialised without the http_client directly provided, and it's used with Langfuse observer decorator,
    we are getting false error messages.

    Unfortunatelly, Langfuse doesn't seem eager to fix this, so this is a workaround. See https://github.com/langfuse/langfuse/issues/5622.

    Use this function as a helper function to create bug-free OpenAIProvider.
    """
    return OpenAIProvider(
        openai_client=AsyncOpenAI(
            api_key=api_key.get_secret_value(),
            base_url=base_url,
            http_client=DefaultAsyncHttpxClient(
                timeout=DEFAULT_TIMEOUT,
                base_url=base_url,
            ),
        )
    )
