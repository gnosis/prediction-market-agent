from crewai import Agent, Crew, Process, Task
from crewai_tools import BaseTool

from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.agents.general_agent.omen_functions import (
    GetBinaryMarketsTool,
)


class GeneralAgent(AbstractAgent):
    def __init__(self) -> None:
        self.gambler = Agent(
            role="Gambler",
            goal="Make money using prediction markets",
            backstory="You are a professional gambler that has access to a range of tools and money, so that you are able to interact with prediction markets in order to increase the amount of money you own.",
            verbose=True,
            allow_delegation=False,
            tools=self.get_tools(),
        )

    def run(self) -> None:
        """
        Method where agent should reason and execute actions on Omen.
        """
        task = self.prepare_task()
        crew = Crew(
            agents=[self.gambler],
            tasks=[task],
            process=Process.sequential,  # Optional: Sequential task execution is default
        )
        result = crew.kickoff(inputs={"topic": "AI in healthcare"})
        print(result)

    def get_tools(self) -> list[BaseTool]:
        return [GetBinaryMarketsTool()]

    def prepare_task(self) -> Task:
        return Task(
            description=(
                "Find 1 or more markets on Omen that you could place bets on."
            ),
            expected_output="A list of 1 or more markets, only displaying the market title.",
            agent=self.gambler,
        )
