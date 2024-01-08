import autogen

from utils import get_keys
from abstract_agent import AbstractAgent


class AutoGenAgent(AbstractAgent):
    def __init__(self):
        keys = get_keys()
        llm_config = {
            "config_list": [
                {
                    "model": "gpt-4",
                    "api_key": keys.openai,
                }
            ],
            "temperature": 0.2,
        }
        self.assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config=llm_config,
        )

        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "")
            .rstrip()
            .endswith("TERMINATE"),
            code_execution_config={"work_dir": ".agents_workspace"},
            llm_config=llm_config,
            system_message="""Reply TERMINATE if the task is completed, otherwise, reply CONTINUE.""",
        )
        # TODO Add functions for web scraping, etc.

    def run(self, objective: str) -> str:
        self.user_proxy.initiate_chat(
            self.assistant,
            message=objective,
        )
        # TODO parse message to remove termination string
        return self.assistant.last_message()
