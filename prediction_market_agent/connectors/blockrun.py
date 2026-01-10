"""
BlockRun LLM Integration for Gnosis Prediction Market Agent.

Uses blockrun-llm SDK for pay-per-request LLM access via x402 micropayments on Base.

Usage:
    from prediction_market_agent.connectors import get_chat_llm

    # Auto-selects BlockRun (if BLOCKRUN_WALLET_KEY set) or OpenAI
    llm = get_chat_llm(model="gpt-4o", temperature=0)
    result = llm.invoke([SystemMessage(...), HumanMessage(...)])
"""

import os
import logging
from typing import List, Dict, Any, Optional, Iterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

logger = logging.getLogger(__name__)


# Model mapping: common model names -> blockrun model names
MODEL_MAP = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o-2024-08-06": "openai/gpt-4o",
    "gpt-4-turbo": "openai/gpt-4o",
    "gpt-4-1106-preview": "openai/gpt-4o",
    "gpt-3.5-turbo-16k": "openai/gpt-4o-mini",
    "claude-3-5-sonnet": "anthropic/claude-sonnet-4",
    "claude-3-5-haiku": "anthropic/claude-haiku-4.5",
    "openai:gpt-4o-2024-08-06": "openai/gpt-4o",
    "openai:gpt-4o": "openai/gpt-4o",
    "openai:gpt-4o-mini": "openai/gpt-4o-mini",
}


class BlockRunChatLLM(BaseChatModel):
    """
    LangChain-compatible Chat LLM using BlockRun x402 payments.

    Uses blockrun-llm SDK (https://github.com/blockrunai/blockrun-llm).
    """

    model: str = "openai/gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 4096
    private_key: Optional[str] = None

    _client: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Normalize model name
        self.model = MODEL_MAP.get(self.model, self.model)

        # Initialize blockrun-llm SDK client
        from blockrun_llm import LLMClient

        key = self.private_key or os.getenv("BLOCKRUN_WALLET_KEY")
        if not key:
            raise ValueError("BLOCKRUN_WALLET_KEY not set")

        self._client = LLMClient(private_key=key)
        logger.info(f"BlockRun LLM initialized: {self.model}")

    @property
    def _llm_type(self) -> str:
        return "blockrun"

    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to blockrun-llm format."""
        result = []
        for msg in messages:
            role = "user"  # default

            # Check class name
            if hasattr(msg, "__class__"):
                class_name = msg.__class__.__name__.lower()
                if "system" in class_name:
                    role = "system"
                elif "human" in class_name or "user" in class_name:
                    role = "user"
                elif "ai" in class_name or "assistant" in class_name:
                    role = "assistant"

            # Check type attribute (LangChain uses this)
            if hasattr(msg, "type"):
                msg_type = msg.type.lower()
                if msg_type == "human":
                    role = "user"
                elif msg_type == "ai":
                    role = "assistant"
                elif msg_type == "system":
                    role = "system"

            content = str(msg.content) if hasattr(msg, "content") else str(msg)
            result.append({"role": role, "content": content})

        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs,
    ) -> ChatResult:
        """Generate response using blockrun-llm SDK."""
        converted_messages = self._convert_messages(messages)

        # Use blockrun-llm SDK's chat_completion method
        result = self._client.chat_completion(
            model=self.model,
            messages=converted_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
        )
        response_text = result.choices[0].message.content
        logger.debug(f"BlockRun response: {response_text[:100]}...")

        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs,
    ) -> Iterator[ChatGeneration]:
        """Streaming not yet supported, falls back to generate."""
        result = self._generate(messages, stop, run_manager, **kwargs)
        yield result.generations[0]


def get_chat_llm(
    model: str = "gpt-4o",
    temperature: float = 0.0,
    openai_api_key: Optional[Any] = None,
    **kwargs,
) -> BaseChatModel:
    """
    Factory function to get the appropriate Chat LLM.

    Auto-selects BlockRun if BLOCKRUN_WALLET_KEY is set, else ChatOpenAI.

    Args:
        model: Model name (e.g., "gpt-4o", "gpt-4o-mini")
        temperature: Sampling temperature
        openai_api_key: OpenAI API key (for fallback)

    Returns:
        BaseChatModel instance
    """
    blockrun_key = os.getenv("BLOCKRUN_WALLET_KEY")

    if blockrun_key:
        try:
            return BlockRunChatLLM(
                model=model,
                temperature=temperature,
                private_key=blockrun_key,
                **kwargs,
            )
        except Exception as e:
            logger.warning(f"BlockRun init failed: {e}, falling back to OpenAI")

    # Fall back to ChatOpenAI
    from langchain_openai import ChatOpenAI

    # Handle pydantic SecretStr
    if openai_api_key and hasattr(openai_api_key, "get_secret_value"):
        openai_api_key = openai_api_key.get_secret_value()

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=openai_api_key,
        **kwargs,
    )
