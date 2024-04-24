"""
Testing out Tavily API methods introduced in https://pypi.org/project/tavily-python/0.2.2/
"""

from enum import Enum

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from prediction_market_agent.utils import APIKeys

keys = APIKeys()
openai_api_key = keys.openai_api_key.get_secret_value()
tavily_api_key = keys.tavily_api_key.get_secret_value()


class Answer(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"

    def is_known(self) -> bool:
        return self != Answer.UNKNOWN

    def to_bool(self) -> bool:
        if self == Answer.UNKNOWN:
            raise ValueError("Cannot convert UNKNOWN to bool")
        return self == Answer.TRUE


tavily_answer_to_answer_prompt = ChatPromptTemplate.from_template(
    template=(
        "Given the following context: {context}\n"
        "Answer the following statement: {statement}\n"
        "Give a one-word answer only: TRUE, FALSE, UNKNOWN."
    ),
)

chain = (
    tavily_answer_to_answer_prompt
    | ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=openai_api_key)
    | StrOutputParser()
)

client = TavilyClient(api_key=tavily_api_key)
q_and_a = {
    "Missouri executed death row inmate Brian Dorsey on 15 April 2024": False,
    "Apple laid off workers in California on 14 April 2024": False,
    "Spotify launched another AI feature on 14 April 2024": False,
    "A new MacBook Air product was released by Apple on 13 April 2024.": False,
    "The solar eclipse was visible from Dallas on 13 April 2024": False,
}

for q, a in q_and_a.items():
    print(f"Question: {q}")
    print(f"Expected answer: {a}")

    answer_from_search = client.qna_search(query=q)
    print(f"Answer from search: {answer_from_search}")

    answer_from_chain = chain.invoke(dict(context=answer_from_search, statement=q))
    answer = Answer(answer_from_chain)
    print(f"Answer from chain: {answer}")

    if answer.is_known():
        print(f"Answer is correct?: {answer.to_bool() == a}")
