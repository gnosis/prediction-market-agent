from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from openai import APIStatusError, AsyncStream
from openai._types import not_given, omit
from openai.types import chat
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from pydantic_ai import ModelHTTPError, UnexpectedModelBehavior
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
    OpenAIChatModelSettings,
    OpenAIModel,
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
        model_settings: OpenAIChatModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> chat.ChatCompletion | AsyncStream[ChatCompletionChunk]:
        tools = self._get_tools(model_request_parameters)

        if not tools:
            tool_choice: Literal["none", "required", "auto"] | None = None
        elif not model_request_parameters.allow_text_output:
            tool_choice = "required"
        else:
            tool_choice = "auto"

        openai_messages = await self._map_messages(messages, model_request_parameters)

        extra_headers = model_settings.get("extra_headers", {})
        extra_headers.setdefault("User-Agent", get_user_agent())

        try:
            return await self.client.chat.completions.create(
                model=self._model_name,
                messages=openai_messages,
                n=self._n,
                parallel_tool_calls=model_settings.get("parallel_tool_calls", omit),
                tools=tools or omit,
                tool_choice=tool_choice or omit,
                stream=stream,
                stream_options={"include_usage": True} if stream else omit,
                stop=model_settings.get("stop_sequences", omit),
                max_completion_tokens=model_settings.get("max_tokens", omit),
                temperature=model_settings.get("temperature", omit),
                top_p=model_settings.get("top_p", omit),
                timeout=model_settings.get("timeout", not_given),
                seed=model_settings.get("seed", omit),
                presence_penalty=model_settings.get("presence_penalty", omit),
                frequency_penalty=model_settings.get("frequency_penalty", omit),
                logit_bias=model_settings.get("logit_bias", omit),
                reasoning_effort=model_settings.get("openai_reasoning_effort", omit),
                user=model_settings.get("openai_user", omit),
                extra_headers=extra_headers,
                extra_body=model_settings.get("extra_body"),
            )
        except APIStatusError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(
                    status_code=status_code, model_name=self.model_name, body=e.body
                ) from e
            raise

    def _process_response(self, response: chat.ChatCompletion | str) -> ModelResponse:
        if not isinstance(response, chat.ChatCompletion):
            raise UnexpectedModelBehavior(
                "Invalid response from OpenAI chat completions endpoint, expected JSON data"
            )

        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        items: list[ModelResponsePart] = []

        for idx, choice in enumerate(response.choices, start=1):
            if choice.message.content is not None:
                items.append(TextPart(f"[variant {idx}] {choice.message.content}"))

            if choice.message.tool_calls:
                for call in choice.message.tool_calls:
                    if isinstance(call, ChatCompletionMessageFunctionToolCall):
                        part = ToolCallPart(
                            call.function.name,
                            call.function.arguments,
                            tool_call_id=call.id,
                        )
                    elif isinstance(
                        call, ChatCompletionMessageCustomToolCall
                    ):  # pragma: no cover
                        # NOTE: Custom tool calls are not supported.
                        # See <https://github.com/pydantic/pydantic-ai/issues/2513> for more details.
                        raise RuntimeError("Custom tool calls are not supported")
                    else:
                        assert False

                    items.append(part)

        return ModelResponse(items, model_name=response.model, timestamp=timestamp)
