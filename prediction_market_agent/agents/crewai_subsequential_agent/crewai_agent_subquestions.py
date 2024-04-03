import json

from langchain_community.tools import DuckDuckGoSearchRun
from prediction_market_agent_tooling.markets.agent_market import AgentMarket

from prediction_market_agent.agents.abstract import AbstractAgent
import crewai
import json
import os
from crewai import Agent
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from pydantic import BaseModel
from crewai import Crew, Process, Agent, Task
from tqdm import tqdm

from prediction_market_agent.agents.crewai_subsequential_agent.prompts import *
search_tool = SerperDevTool()

# TODO can use langchain's @tool decorator on our own tool methods to create a
# tool useable by a crew agent

class Outcomes(BaseModel):
    outcomes: list[str]

class ProbabilityOutput(BaseModel):
    decision: str
    p_yes: float
    p_no: float
    confidence: float

class CrewAIAgent(AbstractAgent):
    def __init__(self) -> None:
        
        self.researcher = Agent(
            role="Research Analyst",
            goal="Research and report on some future event, giving high quality and nuanced analysis",
            backstory="You are a senior research analyst who is adept at researching and reporting on future events.",
            verbose=True,
            allow_delegation=False,
            tools=[search_tool],
        )

        self.analyst = Agent(
        role='Data Analyst',
        goal='Analyze research findings',
        backstory='A meticulous analyst with a knack for uncovering patterns'
        )

        self.writer = Agent(
        role='Writer',
        goal='Draft the final report',
        backstory='A skilled writer with a talent for crafting compelling narratives'
        )

        self.predictor = Agent(
                    role="Professional Gambler",
                    goal="Predict, based on some research you are presented with, whether or not a given event will occur",
                    backstory="You are a professional gambler who is adept at predicting and betting on the outcomes of future events.",
                    verbose=True,
                    allow_delegation=False,
                    tools=[]
                )

    def split_research_into_outcomes(self, question: str) -> Outcomes:
        create_outcomes_task = Task(
          description=(CREATE_OUTCOMES_FROM_SCENARIO_PROMPT),
          expected_output=CREATE_OUTCOMES_FROM_SCENARIO_OUTPUT,
          tools = [],
          output_json=Outcomes,
          agent=self.researcher,
        )

        report_crew = Crew(
            agents=[self.researcher],
            tasks=[create_outcomes_task],
        )
        result = report_crew.kickoff(inputs={'scenario': question})
        return result

    def generate_prediction_for_one_outcome(self, sentence: str) -> ProbabilityOutput:

        task_research_one_outcome = Task(
                    description=(RESEARCH_OUTCOME_PROMPT),
                    tools=[search_tool],
                    agent=self.researcher,
                    expected_output=(RESEARCH_OUTCOME_OUTPUT),
                )
        task_create_probability_for_one_outcome = Task(
                    description=(PROBABILITY_FOR_ONE_OUTCOME_PROMPT),
                expected_output=PROBABILITY_CLASS_OUTPUT,
                    agent=self.predictor,
                output_json=ProbabilityOutput,
            context=[task_research_one_outcome]
                )
        crew = Crew(
                    agents=[self.researcher, self.predictor],
                    tasks=[task_research_one_outcome, task_create_probability_for_one_outcome],
                    verbose=2,
                process=Process.sequential            
                )
        
        result = crew.kickoff(inputs={'sentence': sentence})
        return result

    def generate_final_decision(self, outcomes_with_probabilities) -> ProbabilityOutput:

        task_final_decision = Task(
                    description=(FINAL_DECISION_PROMPT),
                    tools=[],
                    agent=self.predictor,
                    expected_output=(PROBABILITY_CLASS_OUTPUT),
                output_json=ProbabilityOutput,
                )
        
        crew = Crew(
                    agents=[self.predictor],
                    tasks=[task_final_decision],
                    verbose=2, 
                )
        
        result = crew.kickoff(inputs={'outcomes_with_probabilities': 
                                    [(i[0],i[1].dict()) for i in outcomes_with_probabilities],
                                    'number_of_outcomes': len(outcomes_with_probabilities),
                                    'outcome_to_assess': outcomes_with_probabilities[0][0]})
        return result

    def answer_binary_market(self, market: AgentMarket) -> bool:
        
        outcomes = self.split_research_into_outcomes(market.question)

        outcomes_with_probs = []
        for outcome in tqdm(outcomes.outcomes):
            prediction = self.generate_prediction_for_one_outcome(outcome)
            probability_output = ProbabilityOutput.model_validate_json(prediction)
            outcomes_with_probs.append((outcome, probability_output))


        final_answer = self.generate_final_decision(outcomes_with_probs)
        return True if final_answer.decision == "y" else False