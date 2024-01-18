from prediction_market_agent.ai_models.llama_ai_models import (
    construct_llama_prompt,
    LlamaMessage,
    LlamaRole,
)


def test_construct_llama_prompt_0():
    prompt = construct_llama_prompt([])
    assert prompt == ""


def test_construct_llama_prompt_1():
    prompt = construct_llama_prompt(
        [LlamaMessage(role=LlamaRole.user, content="Hello!")]
    )
    assert prompt == "[INST] Hello! [/INST]"


def test_construct_llama_prompt_2():
    prompt = construct_llama_prompt(
        [
            LlamaMessage(role=LlamaRole.user, content="Hello!"),
            LlamaMessage(role=LlamaRole.assistant, content="Bonjour!"),
        ]
    )
    assert prompt == "[INST] Hello! [/INST]\nBonjour!"
