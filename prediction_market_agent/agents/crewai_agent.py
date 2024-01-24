import json
from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.data_models.market_data_models import MarketProtocol

from langchain_community.tools import DuckDuckGoSearchRun

# TODO can use langchain's @tool decorator on our own tool methods to create a
# tool useable by a crew agent


class CrewAIAgent(AbstractAgent):
    def __init__(self) -> None:
        try:
            from crewai import Agent
        except ImportError:
            raise RuntimeError("You need to install the `crewai` package manually.")
        search_tool = DuckDuckGoSearchRun()
        self._researcher = Agent(
            role="Research Analyst",
            goal="Research and report on some future event, giving high quality and nuanced analysis",
            backstory="You are a senior research analyst who is adept at researching and reporting on future events.",
            verbose=True,
            allow_delegation=False,
            tools=[search_tool],
        )
        self._predictor = Agent(
            role="Professional Gambler",
            goal="Predict, based on some research you are presented with, whether or not a given event will occur",
            backstory="You are a professional gambler who is adept at predicting and betting on the outcomes of future events.",
            verbose=True,
            allow_delegation=True,
        )

    def answer_binary_market(self, market: MarketProtocol) -> bool:
        from crewai import Task, Crew

        task1 = Task(
            description=(
                f"Research and report on the following question:\n\n"
                f"{market.question}\n\n"
                f"Search and scrape the web for information that will help you give a high quality, nuanced answer to the question.\n\n"
                f"Return your answer in raw JSON format, with no special formatting such as newlines, as follows:\n\n"
                f'{{"report": <REPORT>}}\n\n'
                f"where <REPORT> is a free text field that contains a well though out justification for the predicition based on the summary of your findings.\n\n"
            ),
            agent=self._researcher,
        )
        task2 = Task(
            description=(
                f"Take the report produced by the 'Research Analyst' and come up with a boolean answer to the the question:\n\n"
                f"{market.question}\n\n"
                f"Return your answer in raw JSON format, with no special formatting such as newlines, as follows:\n\n"
                f'{{"prediction": <PREDICTION>}}\n\n'
                f'where <PREDICTION> is a boolean string (either "True" or "False")'
            ),
            agent=self._predictor,
        )
        crew = Crew(
            agents=[self._researcher, self._predictor],
            tasks=[task1, task2],
            verbose=2,
        )
        result = crew.kickoff()
        report = json.loads(task1.output.result)
        assert "report" in report
        result = json.loads(result)
        assert "prediction" in result
        return True if result["prediction"] == "True" else False
