from prediction_market_agent.ai_models.abstract_ai_models import (
    AbstractAiChatModel,
    Message,
)
from enum import Enum
from typing import Optional, Literal
from prediction_market_agent.tools.utils import should_not_happen
import replicate


class LlamaRole(Enum):
    user = "user"
    assistant = "assistant"


class ChatReplicateLLamaModel(AbstractAiChatModel):
    def __init__(
        self,
        model: Literal["meta/llama-2-70b-chat",] = "meta/llama-2-70b-chat",
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.01,
        top_p: float = 1.0,
        min_new_tokens: int = -1,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.min_new_tokens = min_new_tokens

    def complete(self, messages: list[Message]) -> Optional[str]:
        prompt = construct_llama_prompt(messages)
        completion = "".join(
            replicate.run(
                self.model,
                input={
                    "prompt": prompt,
                    "system_prompt": self.system_prompt or "",
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "min_new_tokens": self.min_new_tokens,
                },
            )
        )
        return completion


def construct_llama_prompt(messages: list[Message]) -> str:
    """
    Based on https://replicate.com/blog/how-to-prompt-llama.
    """
    return "\n".join(
        (
            message.content
            if message.role == LlamaRole.assistant.value
            else f"[INST] {message.content} [/INST]"
            if message.role == LlamaRole.user.value
            else should_not_happen(f"Invalid role in the message: {message}")
        )
        for message in messages
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    model = ChatReplicateLLamaModel()
    messages = [
        Message(role=LlamaRole.user.value, content="Hello!"),
        Message(role=LlamaRole.assistant.value, content="Bonjour!"),
    ]
    completion = model.complete(messages)
    print(completion)
