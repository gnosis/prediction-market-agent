from prediction_market_agent_tooling.deploy.agent import Answer
from pydantic import BaseModel


class Scenarios(BaseModel):
    scenarios: list[str]


class CorrelatedQuestionsAgent:
    def __init__(self, model: str) -> None:
        self.model = model

    def answer_binary_market(self, question: str, n_iterations: int = 1) -> bool:
        """
        ToDo - find correlated questions and based on that answer the question
         Use cosine similarity between questions - https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/#embed_query,
         Use the similarity above to order markets that are closer to the question being asked.
         Then, get probability of those markets as part of the prompt to the agent.

         Markets - fetch from Omen and Manifold
            -> Omen: use filter `title_contains_nocase` and pass in all nouns from questions (but then, how to
            handle Panama and Panamian?)
            Example: questions should be correlated
            -> https://aiomen.eth.limo/#/0x0f6a24228a3da03332ea15d7066879528bb42a5f (Will the Panamanian presidential election result in a clear victor by 12 May 2024?
            -> https://aiomen.eth.limo/#/0xa3a7da3fd71068293df361a42bd206a78c1a2d39 (Will Panama's President-elect Mulino take office on 13 May 2024?)

        """

        return False
