"""Tests for BlockRun LLM connector."""

import os
import pytest
from unittest.mock import patch, MagicMock


def test_get_chat_llm_fallback_to_openai():
    """Test that get_chat_llm falls back to ChatOpenAI when no BLOCKRUN_WALLET_KEY."""
    with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": ""}):
        from prediction_market_agent.connectors import get_chat_llm

        # Mock ChatOpenAI to avoid API key requirement
        with patch("prediction_market_agent.connectors.blockrun.ChatOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            llm = get_chat_llm(model="gpt-4o", temperature=0, openai_api_key="test-key")
            mock_openai.assert_called_once()


def test_get_chat_llm_uses_blockrun():
    """Test that get_chat_llm uses BlockRunChatLLM when BLOCKRUN_WALLET_KEY is set."""
    test_key = "0x0000000000000000000000000000000000000000000000000000000000000001"

    with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": test_key}):
        from prediction_market_agent.connectors import get_chat_llm, BlockRunChatLLM

        llm = get_chat_llm(model="gpt-4o-mini", temperature=0)
        assert isinstance(llm, BlockRunChatLLM)
        assert llm.model == "openai/gpt-4o-mini"  # Model normalized


def test_model_mapping():
    """Test that model names are properly mapped."""
    from prediction_market_agent.connectors.blockrun import MODEL_MAP

    assert MODEL_MAP["gpt-4o"] == "openai/gpt-4o"
    assert MODEL_MAP["gpt-4o-mini"] == "openai/gpt-4o-mini"
    assert MODEL_MAP["openai:gpt-4o-2024-08-06"] == "openai/gpt-4o"


def test_blockrun_llm_type():
    """Test BlockRunChatLLM returns correct type."""
    test_key = "0x0000000000000000000000000000000000000000000000000000000000000001"

    with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": test_key}):
        from prediction_market_agent.connectors import BlockRunChatLLM

        llm = BlockRunChatLLM(model="gpt-4o")
        assert llm._llm_type == "blockrun"
