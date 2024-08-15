import datetime
import typing as t
from abc import ABC

import tenacity
from crewai import Agent, Crew, Process, Task
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from prediction_market_agent_tooling.tools.langfuse_ import observe, langfuse_context
from langchain_core.language_models import BaseChatModel
from langchain_core.pydantic_v1 import SecretStr
from langchain_openai import ChatOpenAI
from openai import APIError
from prediction_market_agent_tooling.deploy.agent import Answer
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.parallelism import (
    DEFAULT_PROCESSPOOL_EXECUTOR,
    par_generator,
    par_map,
)
from prediction_market_agent_tooling.tools.utils import (
    LLM_SUPER_LOW_TEMPERATURE,
    add_utc_timezone_validator,
    utcnow,
)
from pydantic import BaseModel
from requests import HTTPError

from prediction_market_agent.agents.microchain_agent.memory import AnswerWithScenario
from prediction_market_agent.agents.think_thoroughly_agent.models import (
    CorrelatedMarketInput,
)
from prediction_market_agent.agents.think_thoroughly_agent.prompts import (
    CREATE_HYPOTHETICAL_SCENARIOS_FROM_SCENARIO_PROMPT,
    CREATE_REQUIRED_CONDITIONS_PROMPT,
    FINAL_DECISION_PROMPT,
    FINAL_DECISION_WITH_RESEARCH_PROMPT,
    LIST_OF_SCENARIOS_OUTPUT,
    PROBABILITY_CLASS_OUTPUT,
    PROBABILITY_FOR_ONE_OUTCOME_PROMPT,
    RESEARCH_OUTCOME_OUTPUT,
    RESEARCH_OUTCOME_PROMPT,
    RESEARCH_OUTCOME_WITH_PREVIOUS_OUTPUTS_PROMPT,
)
from prediction_market_agent.agents.utils import get_event_date_from_question
from prediction_market_agent.db.long_term_memory_table_handler import (
    LongTermMemoryTableHandler,
)
from prediction_market_agent.db.pinecone_handler import PineconeHandler
from prediction_market_agent.tools.prediction_prophet.research import (
    prophet_make_prediction,
    prophet_research,
    prophet_research_observed,
)
from prediction_market_agent.utils import APIKeys, disable_crewai_telemetry


class Scenarios(BaseModel):
    scenarios: list[str]


class TavilySearchResultsThatWillThrow(TavilySearchResults):
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_fixed(1),
        reraise=True,
    )
    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> tuple[list[dict[str, str]] | str, dict[t.Hashable, t.Any]]:
        """
        Use the tool.
        Throws an exception if it occurs, instead stringifying it.
        """
        raw_results = self.api_wrapper.raw_results(
            query,
            self.max_results,
        )
        return self.api_wrapper.clean_results(raw_results["results"]), raw_results

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_fixed(1),
        reraise=True,
    )
    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> tuple[list[dict[str, str]] | str, dict[t.Hashable, t.Any]]:
        """
        Use the tool asynchronously.
        Throws an exception if it occurs, instead stringifying it.
        """
        raw_results = await self.api_wrapper.raw_results_async(
            query,
            self.max_results,
        )
        return self.api_wrapper.clean_results(raw_results["results"]), raw_results


class ThinkThoroughlyBase(ABC):
    identifier: str

    def __init__(self, model: str, memory: bool = True) -> None:
        self.model = model
        self.subgraph_handler = OmenSubgraphHandler()
        self.pinecone_handler = PineconeHandler()
        self.memory = memory
        self._long_term_memory = (
            LongTermMemoryTableHandler(self.identifier) if self.memory else None
        )

        disable_crewai_telemetry()  # To prevent telemetry from being sent to CrewAI

    @staticmethod
    def _get_current_date() -> str:
        return utcnow().strftime("%Y-%m-%d")

    def save_answer_to_long_term_memory(
        self, answer_with_scenario: AnswerWithScenario
    ) -> None:
        if not self._long_term_memory:
            logger.info(
                "Did not save answer to long term memory since it was not initialized."
            )
            return

        self._long_term_memory.save_answer_with_scenario(answer_with_scenario)

    @staticmethod
    def _get_researcher(model: str, add_langfuse_callback: bool) -> Agent:
        return Agent(
            role="Research Analyst",
            goal="Research and report on some future event, giving high quality and nuanced analysis",
            backstory=f"Current date is {ThinkThoroughlyBase._get_current_date()}. You are a senior research analyst who is adept at researching and reporting on future events.",
            verbose=True,
            allow_delegation=False,
            tools=[ThinkThoroughlyBase._build_tavily_search()],
            llm=ThinkThoroughlyBase._build_llm(model),
            callbacks=(
                [langfuse_context.get_current_langchain_handler()]
                if add_langfuse_callback
                else None
            ),
        )

    @staticmethod
    def _get_predictor(model: str, add_langfuse_callback: bool) -> Agent:
        return Agent(
            role="Professional Gambler",
            goal="Predict, based on some research you are presented with, whether or not a given event will occur",
            backstory=f"Current date is {ThinkThoroughlyBase._get_current_date()}. You are a professional gambler who is adept at predicting and betting on the outcomes of future events.",
            verbose=True,
            allow_delegation=False,
            llm=ThinkThoroughlyBase._build_llm(model),
            callbacks=(
                [langfuse_context.get_current_langchain_handler()]
                if add_langfuse_callback
                else None
            ),
        )

    @staticmethod
    def _build_tavily_search() -> TavilySearchResultsThatWillThrow:
        api_key = SecretStr(APIKeys().tavily_api_key.get_secret_value())
        api_wrapper = TavilySearchAPIWrapper(tavily_api_key=api_key)
        return TavilySearchResultsThatWillThrow(api_wrapper=api_wrapper)

    @staticmethod
    def _build_llm(model: str) -> BaseChatModel:
        keys = APIKeys()
        # ToDo - Add Langfuse callback handler here once integration becomes clear (see
        #  https://github.com/gnosis/prediction-market-agent/issues/107)
        llm = ChatOpenAI(
            model=model,
            api_key=keys.openai_api_key_secretstr_v1,
            temperature=0.0,
        )
        return llm

    def update_markets(self) -> None:
        """We use the agent's run to add embeddings of new markets that don't exist yet in the
        vector DB."""
        created_after = utcnow() - datetime.timedelta(days=7)
        self.pinecone_handler.insert_all_omen_markets_if_not_exists(
            created_after=created_after
        )

    @observe()
    def get_required_conditions(self, question: str) -> Scenarios:
        researcher = self._get_researcher(self.model, add_langfuse_callback=True)

        create_required_conditions = Task(
            description=CREATE_REQUIRED_CONDITIONS_PROMPT,
            expected_output=LIST_OF_SCENARIOS_OUTPUT,
            output_pydantic=Scenarios,
            agent=researcher,
        )

        report_crew = Crew(agents=[researcher], tasks=[create_required_conditions])
        scenarios: Scenarios = report_crew.kickoff(
            inputs={"scenario": question, "n_scenarios": 3}
        )

        logger.info(f"Created conditional scenarios: {scenarios.scenarios}")
        return scenarios

    @observe()
    def get_hypohetical_scenarios(self, question: str) -> Scenarios:
        researcher = self._get_researcher(self.model, add_langfuse_callback=True)

        create_scenarios_task = Task(
            description=CREATE_HYPOTHETICAL_SCENARIOS_FROM_SCENARIO_PROMPT,
            expected_output=LIST_OF_SCENARIOS_OUTPUT,
            output_pydantic=Scenarios,
            agent=researcher,
        )

        report_crew = Crew(agents=[researcher], tasks=[create_scenarios_task])
        scenarios: Scenarios = report_crew.kickoff(
            inputs={"scenario": question, "n_scenarios": 5}
        )

        # Add the original question if it wasn't included by the LLM.
        if question not in scenarios.scenarios:
            scenarios.scenarios.append(question)

        logger.info(f"Created possible hypothetical scenarios: {scenarios.scenarios}")
        return scenarios

    @staticmethod
    def generate_prediction_for_one_outcome(
        model: str,
        scenario: str,
        original_question: str,
        previous_scenarios_and_answers: (
            list[tuple[str, AnswerWithScenario]] | None
        ) = None,
    ) -> AnswerWithScenario | None:
        raise NotImplementedError("This method should be implemented in the subclass.")

    @observe()
    def get_correlated_markets(self, question: str) -> list[CorrelatedMarketInput]:
        nearest_questions = self.pinecone_handler.find_nearest_questions_with_threshold(
            5, text=question
        )

        markets = par_map(
            items=[q.market_address for q in nearest_questions],
            func=lambda market_address: self.subgraph_handler.get_omen_market_by_market_id(
                market_id=market_address
            ),
        )
        return [CorrelatedMarketInput.from_omen_market(market) for market in markets]

    @observe()
    def generate_final_decision(
        self,
        question: str,
        scenarios_with_probabilities: list[t.Tuple[str, AnswerWithScenario]],
        created_time: datetime.datetime | None,
        research_report: str | None = None,
    ) -> Answer:
        predictor = self._get_predictor(self.model, add_langfuse_callback=True)

        task_final_decision = Task(
            description=(
                FINAL_DECISION_PROMPT
                if not research_report
                else FINAL_DECISION_WITH_RESEARCH_PROMPT
            ),
            agent=predictor,
            expected_output=PROBABILITY_CLASS_OUTPUT,
            output_pydantic=Answer,
        )

        crew = Crew(agents=[predictor], tasks=[task_final_decision], verbose=2)

        correlated_markets = self.get_correlated_markets(question)

        event_date = add_utc_timezone_validator(get_event_date_from_question(question))
        n_remaining_days = (event_date - utcnow()).days if event_date else "Unknown"
        n_market_open_days = (
            (utcnow() - add_utc_timezone_validator(created_time)).days
            if created_time
            else "Unknown"
        )
        logger.info(
            f"Event date is {event_date} and {n_remaining_days} days remaining. Market is already open for {n_market_open_days} days."
        )

        logger.info(f"Starting to generate final decision for '{question}'.")
        inputs = {
            "scenarios_with_probabilities": "\n".join(
                f"- Scenario '{s}' has probability of happening {a.p_yes * 100:.2f}% with confidence {a.confidence * 100:.2f}%, because {a.reasoning}"
                for s, a in scenarios_with_probabilities
            ),
            "number_of_scenarios": len(scenarios_with_probabilities),
            "scenario_to_assess": question,
            "correlated_markets": "\n".join(
                f"- Market '{m.question_title}' has {m.current_p_yes * 100:.2f}% probability of happening"
                for m in correlated_markets
            ),
            "n_remaining_days": n_remaining_days,
            "n_market_open_days": n_market_open_days,
        }
        if research_report:
            inputs["research_report"] = research_report
        output: Answer = crew.kickoff(inputs=inputs)
        answer_with_scenario = AnswerWithScenario.build_from_answer(
            output, scenario=question, question=question
        )
        self.save_answer_to_long_term_memory(answer_with_scenario)
        logger.info(
            f"The final prediction is '{output.decision}', with p_yes={output.p_yes}, p_no={output.p_no}, and confidence={output.confidence}"
        )
        return output

    @observe()
    def answer_binary_market(
        self,
        question: str,
        n_iterations: int = 1,
        created_time: datetime.datetime | None = None,
    ) -> Answer | None:
        hypothetical_scenarios = self.get_hypohetical_scenarios(question)
        conditional_scenarios = self.get_required_conditions(question)

        scenarios_with_probs: list[tuple[str, AnswerWithScenario]] = []
        for iteration in range(n_iterations):
            # If n_ierations is > 1, the agent will generate predictions for
            # each scenario multiple times, taking into account the previous
            # predictions. i.e. the probabilities are adjusted iteratively.
            logger.info(
                f"Starting to generate predictions for each scenario, iteration {iteration + 1} / {n_iterations}."
            )

            sub_predictions = par_generator(
                items=[
                    (
                        self.model,
                        scenario,
                        question,
                        scenarios_with_probs,
                        self.generate_prediction_for_one_outcome,
                    )
                    for scenario in (
                        hypothetical_scenarios.scenarios
                        + conditional_scenarios.scenarios
                    )
                ],
                func=process_scenarios,
                executor=DEFAULT_PROCESSPOOL_EXECUTOR,
            )

            scenarios_with_probs = []
            for scenario, prediction in sub_predictions:
                if prediction is None:
                    logger.warning(f"Could not generate prediction for '{scenario}'.")
                    continue
                scenarios_with_probs.append((scenario, prediction))
                logger.info(
                    f"'{scenario}' has prediction {prediction.p_yes * 100:.2f}% chance of being True, because: '{prediction.reasoning}'"
                )
                self.save_answer_to_long_term_memory(prediction)

        final_answer = (
            self.generate_final_decision(
                question, scenarios_with_probs, created_time=created_time
            )
            if scenarios_with_probs
            else None
        )
        if final_answer is None:
            logger.error(
                f"Could not generate final decision for '{question}' with {n_iterations} iterations."
            )
        return final_answer


class ThinkThoroughlyWithItsOwnResearch(ThinkThoroughlyBase):
    identifier = "think-thoroughly-agent"

    @staticmethod
    def generate_prediction_for_one_outcome(
        model: str,
        scenario: str,
        original_question: str,
        previous_scenarios_and_answers: (
            list[tuple[str, AnswerWithScenario]] | None
        ) = None,
    ) -> AnswerWithScenario | None:
        # Do not enable add_langfuse_callback, because it's not thread-safe.
        researcher = ThinkThoroughlyWithItsOwnResearch._get_researcher(
            model, add_langfuse_callback=False
        )
        predictor = ThinkThoroughlyWithItsOwnResearch._get_predictor(
            model, add_langfuse_callback=False
        )

        task_research_one_outcome = Task(
            description=(
                RESEARCH_OUTCOME_PROMPT
                if not previous_scenarios_and_answers
                else RESEARCH_OUTCOME_WITH_PREVIOUS_OUTPUTS_PROMPT
            ),
            agent=researcher,
            expected_output=RESEARCH_OUTCOME_OUTPUT,
        )
        task_create_probability_for_one_outcome = Task(
            description=PROBABILITY_FOR_ONE_OUTCOME_PROMPT,
            expected_output=PROBABILITY_CLASS_OUTPUT,
            agent=predictor,
            output_pydantic=Answer,
            context=[task_research_one_outcome],
        )
        crew = Crew(
            agents=[researcher, predictor],
            tasks=[task_research_one_outcome, task_create_probability_for_one_outcome],
            verbose=2,
            process=Process.sequential,
        )

        inputs = {"sentence": scenario}
        if previous_scenarios_and_answers:
            inputs["previous_scenarios_with_probabilities"] = "\n".join(
                f"- Scenario '{s}' has probability of happening {a.p_yes * 100:.2f}% with confidence {a.confidence * 100:.2f}%, because {a.reasoning}"
                for s, a in previous_scenarios_and_answers
            )

        try:
            output: Answer = crew.kickoff(inputs=inputs)
        except (APIError, HTTPError) as e:
            logger.error(
                f"Could not retrieve response from the model provider because of {e}"
            )
            return None

        if (
            task_research_one_outcome.tools_errors > 0
            or task_create_probability_for_one_outcome.tools_errors > 0
        ):
            logger.error(
                f"Could not retrieve reasonable prediction for '{scenario}' because of errors in the tools"
            )
            return None

        answer_with_scenario = AnswerWithScenario.build_from_answer(
            output, scenario=scenario, question=original_question
        )
        return answer_with_scenario


class ThinkThoroughlyWithPredictionProphetResearch(ThinkThoroughlyBase):
    identifier = "think-thoroughly-prophet-research-agent"

    @staticmethod
    def generate_prediction_for_one_outcome(
        model: str,
        scenario: str,
        original_question: str,
        previous_scenarios_and_answers: (
            list[tuple[str, AnswerWithScenario]] | None
        ) = None,
    ) -> AnswerWithScenario | None:
        if previous_scenarios_and_answers:
            raise ValueError(
                "This agent does not support generating predictions with previous scenarios and answers in mind."
            )

        api_keys = APIKeys()

        try:
            # Don't use observed versions of these functions, because it's running in parallel and Langfuse isn't thread-safe neither process-safe.
            research = prophet_research(
                goal=scenario,
                initial_subqueries_limit=0,  # This agent is making his own subqueries, so we don't need to generate another ones in the research part.
                max_results_per_search=5,
                min_scraped_sites=3,
                model=model,
                openai_api_key=api_keys.openai_api_key,
                tavily_api_key=api_keys.tavily_api_key,
            )
            prediction = prophet_make_prediction(
                market_question=scenario,
                additional_information=research.report,
                engine=model,
                temperature=LLM_SUPER_LOW_TEMPERATURE,
                api_key=api_keys.openai_api_key,
            )
        except Exception as e:
            logger.error(
                f"Could not generate prediction for '{scenario}' because of {e}"
            )
            return None

        if prediction.outcome_prediction is None:
            logger.error(
                f"ThinkThoroughlyWithPredictionProhpetResearch didn't generate prediction for '{scenario}'."
            )
            return None

        return AnswerWithScenario(
            scenario=scenario,
            original_question=original_question,
            decision=prediction.outcome_prediction.decision,
            p_yes=prediction.outcome_prediction.p_yes,
            confidence=prediction.outcome_prediction.confidence,
            reasoning=prediction.outcome_prediction.reasoning,  # TODO: Possible improvement: Prophet currently doesn't return reasoning of its prediction, so it's just None all the time.
        )

    def generate_final_decision(
        self,
        question: str,
        scenarios_with_probabilities: list[t.Tuple[str, AnswerWithScenario]],
        created_time: datetime.datetime | None,
        research_report: str | None = None,
    ) -> Answer:
        api_keys = APIKeys()
        report = (
            research_report
            or prophet_research_observed(
                goal=question,
                model=self.model,
                openai_api_key=api_keys.openai_api_key,
                tavily_api_key=api_keys.tavily_api_key,
            ).report
        )
        return super().generate_final_decision(
            question=question,
            scenarios_with_probabilities=scenarios_with_probabilities,
            created_time=created_time,
            research_report=report,
        )


def process_scenarios(
    inputs: tuple[
        str,
        str,
        str,
        list[tuple[str, AnswerWithScenario]] | None,
        t.Callable[
            [str, str, str, list[tuple[str, AnswerWithScenario]] | None],
            AnswerWithScenario | None,
        ],
    ],
) -> tuple[str, AnswerWithScenario | None]:
    # Needs to be a normal function outside of class, because `lambda` and `self` aren't pickable for processpool executor,
    # and process pool executor is required, because ChromaDB isn't thread-safe.
    # Input arguments needs to be as a single tuple, because par_generator requires a single argument.
    model, scenario, original_question, scenarios_with_probs, process_function = inputs
    return (
        scenario,
        process_function(model, scenario, original_question, scenarios_with_probs),
    )
