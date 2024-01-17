import autogen
import re

from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

from prediction_market_agent import utils
from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.tools.google_search import GoogleSearchTool
from prediction_market_agent.tools.web_scrape import WebScrapingTool


class AutoGenAgent(AbstractAgent):
    def get_base_llm_config(self):
        keys = utils.get_keys()
        return {
            "config_list": [
                {
                    "model": "gpt-4",
                    "api_key": keys.openai,
                }
            ],
            "temperature": 0,
        }

    def __init__(self):
        google_search_tool = GoogleSearchTool()
        web_scaping_tool = WebScrapingTool()
        llm_config = self.get_base_llm_config()
        llm_config["tools"] = [google_search_tool.schema, web_scaping_tool.schema]
        self.assistant = GPTAssistantAgent(
            name="assistant",
            instructions="You are a researcher with tools to search and web scrape, in order to produce high quality, fact-based results for the research objective you've been given. Make sure you search for a variety of high quality sources, and that the results you produce are relevant to the objective you've been given.",
            llm_config=llm_config,
        )
        self.assistant.register_function(
            function_map={
                "web_scraping": web_scaping_tool.fn,
                "google_search": google_search_tool.fn,
            }
        )

        self.user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "")
            .rstrip()
            .endswith("TERMINATE"),
            code_execution_config={"work_dir": ".agents_workspace"},
            llm_config=self.get_base_llm_config(),
            system_message="""Reply TERMINATE if the task is completed, otherwise, reply CONTINUE.""",
        )

    def parse_termination_message_for_result(self, message: str) -> bool:
        pattern = r'\{"result": "(True|False)"\}'
        match = re.search(pattern, message)

        if match:
            return eval(match.group(1))
        else:
            raise ValueError("Result found in Termination message")

    def run(self, market: str) -> str:
        """
        TODO have assistant alsways return in json format with format:
        {
            "reasoning": "<REASONING>",
            "result": "<RESULT>"
        }
        where <REASONING> is a free text field, and <RESULT> is an optional
        boolean field, only set if final answer is known.

        The termination condition is that `result` is set.
        """
        objective = (
            f"Research and report on the following question:\n\n"
            f"{market}\n\n"
            f"Only when the task is completed, your final reponse should contain your reasoning, and as well as the result of your research in the structured json string:\n"
            f'{{"result": "<RESULT>"}}\n'
            f"where <RESULT> is your answer to the question, an can be ONLY either 'True' or 'False'. If you are unsure, make your best guess.\n"
            f"Then finish your reply with \nTERMINATE"
        )
        self.user_proxy.initiate_chat(
            self.assistant,
            message=objective,
        )
        result = self.parse_termination_message_for_result(
            self.assistant.last_message()["content"]
        )
        return result


"""
Example response:
-----------------

assistant (to user_proxy):

From the research gathered through reputable sources, we have the following insights relevant to whether a coalition government will be formed in South Africa following the 2024 general election:

1. The Democratic Alliance (DA) is looking to win the national election and prevent a coalition between the ANC and EFF. However, due to the DA's historical appeal to a mainly white demographic and its track record with previous coalitions, it's uncertain how much impact they will have on forming or preventing a coalition government.

2. The ANC's voter base is declining, and there's a risk it might lose its majority, which brings the possibility of a coalition government into play. Analysts predict the ANC's support could drop below 50 percent, which increases the likelihood of coalition forming. However, the article doesn't explicitly confirm the formation of a coalition but suggests it is a probable outcome.

3. The Economist's analysis indicates that with the ANC potentially winning less than half of the vote, they could lose the majority in parliament. This scenario brings the possibility of a coalition government. It also mentions that the ANC may stay in power if enough votes are gathered, possibly by forming a coalition with smaller parties.

Given these different viewpoints, there seems to be a consensus that the likelihood of a coalition government forming is substantial, especially with the possibility of the ANC not securing an outright majority on its own. Predictions indicate that the ANC might require a coalition with smaller parties to retain power, and the political landscape is such that a coalition government appears to be a potential outcome after the election. 

Therefore, considering the nuances presented in the sources, the answer to whether a coalition government will be formed in South Africa following the 2024 general election is more probable than not, but since we must provide a definitive answer as per the task, our best guess based on the available analysis would be:
{"result": "True"}

TERMINATE

"""
