from enum import Enum
from typing import Literal, Optional

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from prediction_market_agent.ai_models.abstract_ai_models import (
    AbstractAiChatModel,
    Message,
)

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
            "gpt-4o-2024-08-06",
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
        ] = "gpt-4o-2024-08-06",
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
        messages_formatted: list[ChatCompletionMessageParam] = []
        if self.system_prompt is not None:
            messages_formatted.append(
                ChatCompletionSystemMessageParam(
                    role=OpenAiRole.system.value, content=self.system_prompt
                )
            )
        for message in messages:
            if message.role == OpenAiRole.user.value:
                messages_formatted.append(
                    ChatCompletionUserMessageParam(
                        role=message.role, content=message.content  # type: ignore # This is OK due to the if check.
                    )
                )
            else:
                # TODO: Check `ChatCompletionMessageParam` to support all roles, not just hardcoded system and user.
                raise ValueError(
                    f"Only `user` role is supported at the moment, but got `{message.role}`."
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
