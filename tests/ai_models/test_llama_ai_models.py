from prediction_market_agent.ai_models.llama_ai_models import (
    LlamaRole,
    Message,
    construct_llama_prompt,
)


def test_construct_llama_prompt_0() -> None:
    prompt = construct_llama_prompt([])
    assert prompt == ""


def test_construct_llama_prompt_1() -> None:
    prompt = construct_llama_prompt(
        [Message(role=LlamaRole.user.value, content="Hello!")]
    )
    assert prompt == "[INST] Hello! [/INST]"


def test_construct_llama_prompt_2() -> None:
    prompt = construct_llama_prompt(
        [
            Message(role=LlamaRole.user.value, content="Hello!"),
            Message(role=LlamaRole.assistant.value, content="Bonjour!"),
        ]
    )
    assert prompt == "[INST] Hello! [/INST]\nBonjour!"
