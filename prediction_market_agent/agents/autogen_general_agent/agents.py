from typing import Dict, Any

import autogen

from prediction_market_agent.utils import APIKeys


def build_llm_config() -> Dict[str, Any]:
    keys = APIKeys()
    return {
        "config_list": [
            {"model": "gpt-4", "api_key": keys.openai_api_key.get_secret_value()}
        ],
    }


llm_config = build_llm_config()

writer = autogen.AssistantAgent(
    name="Writer",
    llm_config=llm_config,
    system_message="""
    You are a professional writer, known for your insightful and engaging articles.
    You transform complex concepts into compelling narratives.
    You should imporve the quality of the content based on the feedback from the user.
    """,
)

user_proxy = autogen.UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    code_execution_config={
        "last_n_messages": 1,
        "work_dir": "tasks",
        "use_docker": False,
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

critic = autogen.AssistantAgent(
    name="Critic",
    llm_config=llm_config,
    system_message="""
    You are a critic, known for your thoroughness and commitment to standards.
    Your task is to scrutinize content for any harmful elements or regulatory violations, ensuring
    all materials align with required guidelines.
    For code
    """,
)
