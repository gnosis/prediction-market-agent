import autogen
import bs4
import re
import requests
import serpapi
import typing as t

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

from utils import get_keys
from abstract_agent import AbstractAgent


def _google_search(query: str) -> t.List[str]:
    params = {"q": query, "api_key": get_keys().serp, "num": 3}
    search = serpapi.GoogleSearch(params)
    urls = [result["link"] for result in search.get_dict()["organic_results"]]
    return urls


def _summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )
    docs = text_splitter.create_documents([content])
    map_prompt = (
        "Write a summary of the following text for {objective}:\n"
        '"{text}\n'
        "SUMMARY:"
    )
    t = PromptTemplate(template=map_prompt, input_variables=["text", "objective"])
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=t,
        combine_prompt=t,
        verbose=False,
    )
    return summary_chain.run(input_documents=docs, objective=objective)


def _web_scraping(objective: str, url: str):
    response = requests.get(url)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.content, "html.parser")
    text = soup.get_text()
    if len(text) > 10000:
        text = _summary(objective, text)
    return text


class AutoGenAgent(AbstractAgent):
    def get_base_llm_config(self):
        keys = get_keys()
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
        def google_search(query) -> t.List[str]:
            return _google_search(query)

        google_search_schema = {
            "type": "function",
            "function": {
                "name": "google_search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The google search query.",
                        }
                    },
                    "required": ["query"],
                },
                "description": "Google search to return search results from a query.",
            },
        }

        def web_scraping(objective, url) -> str:
            return _web_scraping(objective, url)

        web_scraping_schema = {
            "type": "function",
            "function": {
                "name": "web_scraping",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "objective": {
                            "type": "string",
                            "description": "The objective that defines the content to be scraped from the website.",
                        },
                        "url": {
                            "type": "string",
                            "description": "The URL of the website to be scraped.",
                        },
                    },
                    "required": ["query"],
                },
                "description": "Google search to return search results from a query.",
            },
        }

        llm_config = self.get_base_llm_config()
        llm_config["tools"] = [google_search_schema, web_scraping_schema]
        self.assistant = GPTAssistantAgent(
            name="assistant",
            instructions="You are a researcher with tools to search and web scrape, in order to produce high quality, fact-based results for the research objective you've been given. Make sure you search for a variety of high quality sources, and that the results you produce are relevant to the objective you've been given.",
            llm_config=llm_config,
        )
        self.assistant.register_function(
            function_map={
                "web_scraping": web_scraping,
                "google_search": google_search,
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
            return bool(match.group(1))
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

if __name__ == "__main__":
    # Test util functions of agent
    objective = "In 2024, will Israel conduct a military strike against Iran?"
    keys = get_keys()
    search = _google_search(query=objective, api_key=keys.serp)
    scraping = _web_scraping(objective, search[0])
    summary = _summary(objective, scraping)
    print(scraping)
