from crewai import Agent, Task, Process, Crew
from crewai_tools import SerperDevTool
from langchain_community.callbacks.manager import get_openai_callback
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from pydantic import BaseModel
from tqdm import tqdm
import typing as t

from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.agents.crewai_subsequential_agent.prompts import *

search_tool = SerperDevTool()


class Outcomes(BaseModel):
    outcomes: list[str]


class ProbabilityOutput(BaseModel):
    decision: str
    p_yes: float
    p_no: float
    confidence: float


get_openai_callback()


class CrewAIAgentSubquestions(AbstractAgent):
    def __init__(self) -> None:
        self.researcher = Agent(
            role="Research Analyst",
            goal="Research and report on some future event, giving high quality and nuanced analysis",
            backstory="You are a senior research analyst who is adept at researching and reporting on future events.",
            verbose=True,
            allow_delegation=False,
            tools=[search_tool],
        )

        self.predictor = Agent(
            role="Professional Gambler",
            goal="Predict, based on some research you are presented with, whether or not a given event will occur",
            backstory="You are a professional gambler who is adept at predicting and betting on the outcomes of future events.",
            verbose=True,
            allow_delegation=False,
        )

    def split_research_into_outcomes(self, question: str) -> Outcomes:
        create_outcomes_task = Task(
            description=CREATE_OUTCOMES_FROM_SCENARIO_PROMPT,
            expected_output=CREATE_OUTCOMES_FROM_SCENARIO_OUTPUT,
            output_json=Outcomes,
            agent=self.researcher,
        )

        report_crew = Crew(
            agents=[self.researcher],
            tasks=[create_outcomes_task],
        )
        result = report_crew.kickoff(inputs={'scenario': question})
        return Outcomes.model_validate_json(result)

    def build_tasks_for_outcome(self, input_dict: dict[str, t.Any] = {}) -> list[Task]:
        task_research_one_outcome = Task(
            description=RESEARCH_OUTCOME_PROMPT.format(**input_dict),
            agent=self.researcher,
            expected_output=RESEARCH_OUTCOME_OUTPUT,
            async_execution=True
        )
        task_create_probability_for_one_outcome = Task(
            description=PROBABILITY_FOR_ONE_OUTCOME_PROMPT,
            expected_output=PROBABILITY_CLASS_OUTPUT,
            agent=self.predictor,
            output_json=ProbabilityOutput,
            async_execution=True,
            context=[task_research_one_outcome]
        )

        return [task_research_one_outcome, task_create_probability_for_one_outcome]

    def generate_prediction_for_one_outcome(self, sentence: str) -> ProbabilityOutput:
        task_research_one_outcome = Task(
            description=RESEARCH_OUTCOME_PROMPT,
            agent=self.researcher,
            expected_output=RESEARCH_OUTCOME_OUTPUT,
        )
        task_create_probability_for_one_outcome = Task(
            description=PROBABILITY_FOR_ONE_OUTCOME_PROMPT,
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
        return ProbabilityOutput.model_validate_json(result)

    def generate_final_decision(self, outcomes_with_probabilities) -> ProbabilityOutput:
        task_final_decision = Task(
            description=(FINAL_DECISION_PROMPT),
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
                                          [(i[0], i[1].dict()) for i in outcomes_with_probabilities],
                                      'number_of_outcomes': len(outcomes_with_probabilities),
                                      'outcome_to_assess': outcomes_with_probabilities[0][0]})
        return ProbabilityOutput.model_validate_json(result)

    def answer_binary_market(self, market: AgentMarket) -> bool:

        outcomes = self.split_research_into_outcomes(market.question)
        print ("outcomes ", outcomes)

        outcomes_with_probs = []
        task_map = {}
        for outcome in tqdm(outcomes.outcomes):
            tasks_for_outcome = self.build_tasks_for_outcome(input_dict={"sentence": outcome})
            task_map[outcome] = tasks_for_outcome

        # flatten nested list
        all_tasks = sum(task_map.values(), [])
        crew = Crew(
            agents=[self.researcher, self.predictor],
            tasks=all_tasks,
            verbose=2,
        )

        crew.kickoff()

        # We parse individual task results to build outcomes_with_probs
        for outcome, tasks in task_map.items():
            try:
                prediction_result = ProbabilityOutput.model_validate_json(tasks[1].output.raw_output)
            except Exception as e:
                print("Could not parse result as ProbabilityOutput ", e)
                prediction_result = ProbabilityOutput(p_yes=0.5, p_no=0.5, confidence=0, decision="")

            outcomes_with_probs.append((outcome, prediction_result))

        final_answer = self.generate_final_decision(outcomes_with_probs)
        return True if final_answer.decision == "y" else False
