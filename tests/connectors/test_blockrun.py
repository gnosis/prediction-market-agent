"""Tests for BlockRun LLM connector."""

import os
import pytest
from unittest.mock import patch, MagicMock


def _make_mock_client():
    """Return a MagicMock that satisfies LLMClient usage."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = "test response"
    client.chat_completion.return_value = MagicMock(choices=[choice])
    return client


def test_get_chat_llm_fallback_to_openai():
    """Test that get_chat_llm falls back to ChatOpenAI when no BLOCKRUN_WALLET_KEY."""
    with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": ""}):
        from prediction_market_agent.connectors import get_chat_llm

        with patch("langchain_openai.ChatOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            llm = get_chat_llm(model="gpt-4o", temperature=0, openai_api_key="test-key")
            mock_openai.assert_called_once()


def test_get_chat_llm_uses_blockrun():
    """Test that get_chat_llm uses BlockRunChatLLM when BLOCKRUN_WALLET_KEY is set."""
    test_key = "0x0000000000000000000000000000000000000000000000000000000000000001"
    mock_client = _make_mock_client()

    with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": test_key}):
        with patch("blockrun_llm.LLMClient", return_value=mock_client):
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
    assert MODEL_MAP["gpt-4.1-mini-2025-04-14"] == "openai/gpt-4o-mini"


def test_blockrun_llm_type():
    """Test BlockRunChatLLM returns correct type and clears private_key after init."""
    test_key = "0x0000000000000000000000000000000000000000000000000000000000000001"
    mock_client = _make_mock_client()

    with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": test_key}):
        with patch("blockrun_llm.LLMClient", return_value=mock_client):
            from prediction_market_agent.connectors import BlockRunChatLLM

            llm = BlockRunChatLLM(model="gpt-4o")
            assert llm._llm_type == "blockrun"
            assert llm.private_key is None  # Must be cleared after init


def test_blockrun_model_remap_logging(caplog):
    """Test that model remapping is logged."""
    import logging
    test_key = "0x0000000000000000000000000000000000000000000000000000000000000001"
    mock_client = _make_mock_client()

    with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": test_key}):
        with patch("blockrun_llm.LLMClient", return_value=mock_client):
            from prediction_market_agent.connectors import BlockRunChatLLM

            with caplog.at_level(logging.INFO, logger="prediction_market_agent.connectors.blockrun"):
                BlockRunChatLLM(model="gpt-4-turbo")

            assert any("remapped" in r.message for r in caplog.records)


def test_blockrun_choices_empty_raises():
    """Test that empty choices list raises EmptyResponseError."""
    test_key = "0x0000000000000000000000000000000000000000000000000000000000000001"
    mock_client = MagicMock()
    mock_client.chat_completion.return_value = MagicMock(choices=[])

    with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": test_key}):
        with patch("blockrun_llm.LLMClient", return_value=mock_client):
            from prediction_market_agent.connectors.blockrun import BlockRunChatLLM, EmptyResponseError
            from langchain_core.messages import HumanMessage

            llm = BlockRunChatLLM(model="gpt-4o")
            with pytest.raises(EmptyResponseError, match="no choices"):
                llm._generate([HumanMessage(content="hi")])


def test_blockrun_api_error_raises_runtime_error():
    """Test that API errors are wrapped in RuntimeError."""
    test_key = "0x0000000000000000000000000000000000000000000000000000000000000001"
    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = ConnectionError("timeout")

    with patch.dict(os.environ, {"BLOCKRUN_WALLET_KEY": test_key}):
        with patch("blockrun_llm.LLMClient", return_value=mock_client):
            from prediction_market_agent.connectors import BlockRunChatLLM
            from langchain_core.messages import HumanMessage

            llm = BlockRunChatLLM(model="gpt-4o")
            with pytest.raises(RuntimeError, match="BlockRun API call failed"):
                llm._generate([HumanMessage(content="hi")])
