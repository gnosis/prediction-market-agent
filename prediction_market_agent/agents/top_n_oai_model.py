from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from openai import NOT_GIVEN, APIStatusError, AsyncStream
from openai.types import chat
from openai.types.chat import ChatCompletionChunk
from pydantic_ai import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import get_user_agent
from pydantic_ai.models.openai import (
    ModelRequestParameters,
    OpenAIModel,
    OpenAIModelSettings,
)


# TODO: Remove this after https://github.com/pydantic/pydantic-ai/issues/2003 is released in lib
class TopNOpenAINModel(OpenAIModel):
    """OpenAI model wrapper that can ask for N alternative completions."""

    def __init__(self, model_name: str, *, n: int = 3, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self._n = n

    async def _completions_create(  # type: ignore[override]
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)

        if not tools:
            tool_choice: Literal["none", "required", "auto"] | None = None
        elif not model_request_parameters.allow_text_output:
            tool_choice = "required"
        else:
            tool_choice = "auto"

        openai_messages = await self._map_messages(messages)

        extra_headers = model_settings.get("extra_headers", {})
        extra_headers.setdefault("User-Agent", get_user_agent())

        try:
            return await self.client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                n=self._n,
                parallel_tool_calls=model_settings.get(
                    "parallel_tool_calls", NOT_GIVEN
                ),
                tools=tools or NOT_GIVEN,
                tool_choice=tool_choice or NOT_GIVEN,
                stream=stream,
                stream_options={"include_usage": True} if stream else NOT_GIVEN,
                stop=model_settings.get("stop_sequences", NOT_GIVEN),
                max_completion_tokens=model_settings.get("max_tokens", NOT_GIVEN),
                temperature=model_settings.get("temperature", NOT_GIVEN),
                top_p=model_settings.get("top_p", NOT_GIVEN),
                timeout=model_settings.get("timeout", NOT_GIVEN),
                seed=model_settings.get("seed", NOT_GIVEN),
                presence_penalty=model_settings.get("presence_penalty", NOT_GIVEN),
                frequency_penalty=model_settings.get("frequency_penalty", NOT_GIVEN),
                logit_bias=model_settings.get("logit_bias", NOT_GIVEN),
                reasoning_effort=model_settings.get(
                    "openai_reasoning_effort", NOT_GIVEN
                ),
                user=model_settings.get("openai_user", NOT_GIVEN),
                extra_headers=extra_headers,
                extra_body=model_settings.get("extra_body"),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(
                    status_code=status_code, model_name=self.model_name, body=e.body
                ) from e
            raise

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        items: list[ModelResponsePart] = []

        for idx, choice in enumerate(response.choices, start=1):
            if choice.message.content is not None:
                items.append(TextPart(f"[variant {idx}] {choice.message.content}"))

            if choice.message.tool_calls:
                for call in choice.message.tool_calls:
                    items.append(
                        ToolCallPart(
                            call.function.name,
                            call.function.arguments,
                            tool_call_id=call.id,
                        )
                    )

        return ModelResponse(items, model_name=response.model, timestamp=timestamp)
