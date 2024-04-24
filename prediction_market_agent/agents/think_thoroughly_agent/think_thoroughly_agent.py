import typing as t

from crewai import Agent, Crew, Process, Task
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger
from prediction_market_agent_tooling.monitor.langfuse.langfuse_wrapper import (
    LangfuseWrapper,
)
from pydantic import BaseModel

from prediction_market_agent.agents.think_thoroughly_agent.prompts import (
    CREATE_OUTCOMES_FROM_SCENARIO_OUTPUT,
    CREATE_OUTCOMES_FROM_SCENARIO_PROMPT,
    FINAL_DECISION_PROMPT,
    PROBABILITY_CLASS_OUTPUT,
    PROBABILITY_FOR_ONE_OUTCOME_PROMPT,
    RESEARCH_OUTCOME_OUTPUT,
    RESEARCH_OUTCOME_PROMPT,
)
from prediction_market_agent.tools.custom_crewai_tools import TavilyDevTool
from prediction_market_agent.utils import APIKeys

tavily_search = TavilyDevTool()


class Outcomes(BaseModel):
    outcomes: list[str]


class ProbabilityOutput(BaseModel):
    decision: str
    p_yes: float
    p_no: float
    confidence: float


class CrewAIAgentSubquestions:
    def __init__(self, langfuse_wrapper: LangfuseWrapper) -> None:
        llm = self._build_llm(langfuse_wrapper)
        self.researcher = Agent(
            role="Research Analyst",
            goal="Research and report on some future event, giving high quality and nuanced analysis",
            backstory="You are a senior research analyst who is adept at researching and reporting on future events.",
            verbose=True,
            allow_delegation=False,
            tools=[tavily_search],
            llm=llm,
        )

        self.predictor = Agent(
            role="Professional Gambler",
            goal="Predict, based on some research you are presented with, whether or not a given event will occur",
            backstory="You are a professional gambler who is adept at predicting and betting on the outcomes of future events.",
            verbose=True,
            allow_delegation=False,
            llm=llm,
        )

    def _build_llm(self, langfuse_wrapper: LangfuseWrapper) -> BaseChatModel:
        keys = APIKeys()
        llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            api_key=keys.openai_api_key.get_secret_value(),
            callbacks=[langfuse_wrapper.get_langfuse_handler()],
        )
        return llm

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
        result = report_crew.kickoff(inputs={"scenario": question})
        outcomes = Outcomes.model_validate_json(result)
        logger.info(f"Created possible outcomes: {outcomes.outcomes}")
        return outcomes

    def build_tasks_for_outcome(self, input_dict: dict[str, t.Any] = {}) -> list[Task]:
        task_research_one_outcome = Task(
            description=RESEARCH_OUTCOME_PROMPT.format(**input_dict),
            agent=self.researcher,
            expected_output=RESEARCH_OUTCOME_OUTPUT,
            async_execution=True,
        )
        task_create_probability_for_one_outcome = Task(
            description=PROBABILITY_FOR_ONE_OUTCOME_PROMPT,
            expected_output=PROBABILITY_CLASS_OUTPUT,
            agent=self.predictor,
            output_json=ProbabilityOutput,
            async_execution=False,
            context=[task_research_one_outcome],
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
            context=[task_research_one_outcome],
        )
        crew = Crew(
            agents=[self.researcher, self.predictor],
            tasks=[task_research_one_outcome, task_create_probability_for_one_outcome],
            verbose=2,
            process=Process.sequential,
        )

        result = crew.kickoff(inputs={"sentence": sentence})
        output = ProbabilityOutput.model_validate_json(result)
        logger.info(
            f"For the sentence '{sentence}', the prediction is '{output.decision}', with p_yes={output.p_yes}, p_no={output.p_no}, and confidence={output.confidence}"
        )
        return output

    def generate_final_decision(
        self, outcomes_with_probabilities: list[t.Tuple[str, ProbabilityOutput]]
    ) -> ProbabilityOutput:
        task_final_decision = Task(
            description=FINAL_DECISION_PROMPT,
            agent=self.predictor,
            expected_output=PROBABILITY_CLASS_OUTPUT,
            output_json=ProbabilityOutput,
        )

        crew = Crew(
            agents=[self.predictor],
            tasks=[task_final_decision],
            verbose=2,
        )

        crew.kickoff(
            inputs={
                "outcomes_with_probabilities": [
                    (i[0], i[1].model_dump()) for i in outcomes_with_probabilities
                ],
                "number_of_outcomes": len(outcomes_with_probabilities),
                "outcome_to_assess": outcomes_with_probabilities[0][0],
            }
        )
        output = ProbabilityOutput.model_validate_json(
            task_final_decision.output.raw_output
        )
        logger.info(
            f"The final prediction is '{output.decision}', with p_yes={output.p_yes}, p_no={output.p_no}, and confidence={output.confidence}"
        )
        return output

    def answer_binary_market(self, question: str) -> ProbabilityOutput:
        outcomes = self.split_research_into_outcomes(question)

        outcomes_with_probs = []
        task_map = {}
        for outcome in outcomes.outcomes:
            tasks_for_outcome = self.build_tasks_for_outcome(
                input_dict={"sentence": outcome}
            )
            task_map[outcome] = tasks_for_outcome

        # flatten nested list
        all_tasks = sum(task_map.values(), [])
        crew = Crew(
            agents=[self.researcher, self.predictor],
            tasks=all_tasks,
            verbose=2,
            process=Process.sequential,
        )

        # crew.kickoff doesn't finish all async tasks when done.
        crew.kickoff()

        # We parse individual task results to build outcomes_with_probs
        for outcome, tasks in task_map.items():
            raw_output = tasks[1].output.raw_output
            try:
                prediction_result = ProbabilityOutput.model_validate_json(raw_output)
            except Exception as e:
                logger.error(
                    f"Could not parse the result ('{raw_output}') as ProbabilityOutput because of {e}"
                )
                prediction_result = ProbabilityOutput(
                    p_yes=0.5, p_no=0.5, confidence=0, decision=""
                )

            outcomes_with_probs.append((outcome, prediction_result))

        final_answer = self.generate_final_decision(outcomes_with_probs)
        return final_answer
