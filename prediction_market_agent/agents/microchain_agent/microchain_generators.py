import typing as t
from enum import Enum

from microchain.models.generators import TokenTracker


class Llama31SupportedRole(str, Enum):
    system = "system"
    assistant = "assistant"
    user = "user"
    ipython = "ipython"


class Llama31Message(t.TypedDict):
    role: Llama31SupportedRole
    content: str


MESSAGE_BLOCK_TEMPLATE = """<|start_header_id|>{role}<|end_header_id|>

{content}

You are a helpful assistant<|eot_id|>"""


def verify_system_message_is_first(messages: list[Llama31Message]) -> None:
    if any(message["role"] == Llama31SupportedRole.system for message in messages[1:]):
        raise ValueError(
            f"System message should be the first message in the conversation."
        )


def format_llama31_prompt(messages: list[Llama31Message]) -> str:
    """
    Format the messages into a prompt for the Llama 3.1 model,
    based on https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1.
    """
    verify_system_message_is_first(messages)

    prompt = """<|begin_of_text|>"""

    for message in messages:
        prompt += MESSAGE_BLOCK_TEMPLATE.format(**message)

    prompt += "<|start_header_id|>assistant<|end_header_id|>"

    return prompt


def limit_messages(
    messages: list[Llama31Message], max_messages: int
) -> list[Llama31Message]:
    """
    Limit the number of messages in the conversation to `max_messages` and the system message, if included.
    """
    verify_system_message_is_first(messages)
    if messages and messages[0]["role"] == Llama31SupportedRole.system:
        system_message = [messages[0]]
        messages = messages[1:]
    else:
        system_message = []

    return system_message + messages[-max_messages:]


class ReplicateLlama31:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        # Default values set according to their API: https://replicate.com/meta/meta-llama-3.1-405b-instruct
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 1024,
        token_tracker: TokenTracker | None = None,
    ) -> None:
        try:
            from replicate.client import Client
        except ImportError:
            raise ImportError("Please install replicate using pip install replicate")

        self.model = model
        self.client = Client(api_token=api_key)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens
        self.token_tracker = token_tracker
        assert (
            self.token_tracker is None
        ), "Token tracker not supported for Replicate yet."

    def __call__(
        self, messages: list[Llama31Message], stop: list[str] | None = None
    ) -> str:
        prompt = format_llama31_prompt(messages)
        completion = self.client.predictions.create(
            model=self.model,
            input={
                "prompt": prompt,
                "prompt_template": "{prompt}",  # Force Replicate's API to just use our prompt as-is, otherwise they would use their default formatting which doesn't work for list of messages.
                "stop": stop,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            stream=True,
        )
        output = "".join(str(event) for event in completion.stream()).strip()

        if self.token_tracker:
            raise NotImplementedError("Token tracker not supported for Replicate yet.")

        return output

    def print_usage(self) -> None:
        if self.token_tracker:
            print(
                f"Usage: prompt={self.token_tracker.prompt_tokens}, completion={self.token_tracker.completion_tokens}, cost=${self.token_tracker.get_total_cost(self.model):.2f}"
            )
        else:
            print("Token tracker not available")
