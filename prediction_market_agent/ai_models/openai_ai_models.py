from prediction_market_agent.ai_models.abstract_ai_models import (
    AbstractAiChatModel,
    Message,
)
from enum import Enum
from openai import OpenAI
from typing import Optional, Literal
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

ROLE_KEY = "role"
CONTENT_KEY = "content"


class OpenAiRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatOpenAIModel(AbstractAiChatModel):
    def __init__(
        self,
        model: Literal[
            "gpt-4-1106-preview",
            "gpt-4-vision-preview",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-32k",
            "gpt-4-32k-0314",
            "gpt-4-32k-0613",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k-0613",
        ] = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.client = OpenAI(api_key=api_key)

    def complete(self, messages: list[Message]) -> Optional[str]:
        # TODO: Check `ChatCompletionMessageParam` to support all roles, not just hardcoded system and user.
        messages_formatted: list[dict[str, str]] = []
        if self.system_prompt is not None:
            messages_formatted.append(
                {
                    ROLE_KEY: OpenAiRole.system.value,
                    CONTENT_KEY: self.system_prompt,
                }
            )
        for message in messages:
            messages_formatted.append(
                {ROLE_KEY: message.role, CONTENT_KEY: message.content}
            )
        response: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=messages_formatted,
            n=1,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
        )
        completion = response.choices[0].message.content
        return completion


if __name__ == "__main__":
    model = ChatOpenAIModel()
    messages = [Message(role="user", content="Hello, how are you?")]
    completion = model.complete(messages)
    print(completion)
