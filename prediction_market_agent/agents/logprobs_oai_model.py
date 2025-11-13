from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Annotated, Any, Union

import pydantic
from openai.types import chat
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


@dataclass
class LogProbsModelResponse(ModelResponse):
    vendor_details: dict[str, Any] | None = field(default=None, repr=False)


ModelMessage = Annotated[
    Union[ModelRequest, ModelResponse, LogProbsModelResponse],
    pydantic.Discriminator("kind"),
]


class LogProbsOpenAIModel(OpenAIModel):
    def __init__(self, model: str, provider: OpenAIProvider):
        super().__init__(model, provider=provider)

    def _process_response(self, response: chat.ChatCompletion | str) -> ModelResponse:
        if not isinstance(response, chat.ChatCompletion):
            raise UnexpectedModelBehavior(
                "Invalid response from OpenAI chat completions endpoint, expected JSON data"
            )

        timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        choice = response.choices[0]
        items: list[ModelResponsePart] = []
        vendor_details: dict[str, Any] | None = None

        # Add logprobs to vendor_details if available
        if choice.logprobs is not None and choice.logprobs.content:
            # Convert logprobs to a serializable format
            vendor_details = {
                "logprobs": [
                    {
                        "token": lp.token,
                        "bytes": lp.bytes,
                        "logprob": lp.logprob,
                        "top_logprobs": [
                            {
                                "token": tlp.token,
                                "bytes": tlp.bytes,
                                "logprob": tlp.logprob,
                            }
                            for tlp in lp.top_logprobs
                        ],
                    }
                    for lp in choice.logprobs.content
                ],
            }
        if choice.message.content is not None:
            items.append(TextPart(choice.message.content))
        if choice.message.tool_calls is not None:
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
        return LogProbsModelResponse(
            items,
            model_name=response.model,
            timestamp=timestamp,
            vendor_details=vendor_details,
        )
