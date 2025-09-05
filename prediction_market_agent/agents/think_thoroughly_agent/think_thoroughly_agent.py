import typing as t
from abc import ABC
from uuid import UUID, uuid4

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.tools import tool
from prediction_market_agent_tooling.deploy.agent import initialize_langfuse
from prediction_market_agent_tooling.loggers import logger, patch_logger
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.langfuse_ import langfuse_context, observe
from prediction_market_agent_tooling.tools.openai_utils import (
    OpenAIModel,
    get_openai_provider,
)
from prediction_market_agent_tooling.tools.parallelism import par_generator, par_map
from prediction_market_agent_tooling.tools.tavily.tavily_search import tavily_search
from prediction_market_agent_tooling.tools.utils import (
    LLM_SUPER_LOW_TEMPERATURE,
    DatetimeUTC,
    utcnow,
)
from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAIAgent
from pydantic_ai.models import KnownModelName
from pydantic_ai.settings import ModelSettings

from prediction_market_agent.agents.identifiers import (
    THINK_THOROUGHLY,
    THINK_THOROUGHLY_PROPHET,
    AgentIdentifier,
)
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
)
from prediction_market_agent.utils import APIKeys, disable_crewai_telemetry


class Scenarios(BaseModel):
    scenarios: list[str]


@tool
@observe()
def tavily_search_tool(query: str) -> list[dict[str, str]]:
    """
    Given a search query, returns a list of dictionaries with results from internet search using Tavily.
    """
    output = tavily_search(query=query)
    return [
        {
            "title": r.title,
            "url": r.url,
            "content": r.content,
        }
        for r in output.results
    ]


class ThinkThoroughlyBase(ABC):
    identifier: AgentIdentifier
    model: KnownModelName
    model_for_generate_prediction_for_one_outcome: KnownModelName

    def __init__(self, enable_langfuse: bool, memory: bool = True) -> None:
        self.enable_langfuse = enable_langfuse
        self.subgraph_handler = OmenSubgraphHandler()
        self.pinecone_handler = PineconeHandler()
        self.memory = memory
        self._long_term_memory = (
            LongTermMemoryTableHandler.from_agent_identifier(self.identifier)
            if self.memory
            else None
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
    def _get_researcher(model: str) -> Agent:
        langfuse_callback = langfuse_context.get_current_langchain_handler()
        return Agent(
            role="Research Analyst",
            goal="Research and report on some future event, giving high quality and nuanced analysis",
            backstory=f"Current date is {ThinkThoroughlyBase._get_current_date()}. You are a senior research analyst who is adept at researching and reporting on future events.",
            verbose=True,
            allow_delegation=False,
            tools=[tavily_search_tool],
            llm=ThinkThoroughlyBase._build_llm(model),
            callbacks=[langfuse_callback] if langfuse_callback else None,
        )

    @staticmethod
    def _get_predictor(model: str) -> Agent:
        langfuse_callback = langfuse_context.get_current_langchain_handler()
        return Agent(
            role="Professional Gambler",
            goal="Predict, based on some research you are presented with, whether or not a given event will occur",
            backstory=f"Current date is {ThinkThoroughlyBase._get_current_date()}. You are a professional gambler who is adept at predicting and betting on the outcomes of future events.",
            verbose=True,
            allow_delegation=False,
            llm=ThinkThoroughlyBase._build_llm(model),
            callbacks=[langfuse_callback] if langfuse_callback else None,
        )

    @staticmethod
    def _build_llm(model: str) -> LLM:
        keys = APIKeys()
        # ToDo - Add Langfuse callback handler here once integration becomes clear (see
        #  https://github.com/gnosis/prediction-market-agent/issues/107)
        llm = LLM(
            model=model,
            api_key=keys.openai_api_key.get_secret_value(),
            temperature=0,
        )
        return llm

    @observe()
    def get_required_conditions(self, question: str) -> Scenarios:
        researcher = self._get_researcher(self.model)

        create_required_conditions = Task(
            description=CREATE_REQUIRED_CONDITIONS_PROMPT,
            expected_output=LIST_OF_SCENARIOS_OUTPUT,
            output_pydantic=Scenarios,
            agent=researcher,
        )

        report_crew = Crew(agents=[researcher], tasks=[create_required_conditions])
        output = report_crew.kickoff(inputs={"scenario": question, "n_scenarios": 3})
        scenarios: Scenarios = output.pydantic

        logger.info(f"Created conditional scenarios: {scenarios.scenarios}")
        return scenarios

    @observe()
    def get_hypohetical_scenarios(self, question: str) -> Scenarios:
        researcher = self._get_researcher(self.model)

        create_scenarios_task = Task(
            description=CREATE_HYPOTHETICAL_SCENARIOS_FROM_SCENARIO_PROMPT,
            expected_output=LIST_OF_SCENARIOS_OUTPUT,
            output_pydantic=Scenarios,
            agent=researcher,
        )

        report_crew = Crew(agents=[researcher], tasks=[create_scenarios_task])
        output = report_crew.kickoff(inputs={"scenario": question, "n_scenarios": 5})
        scenarios: Scenarios = output.pydantic

        # Add the original question if it wasn't included by the LLM.
        if question not in scenarios.scenarios:
            scenarios.scenarios.append(question)

        logger.info(f"Created possible hypothetical scenarios: {scenarios.scenarios}")
        return scenarios

    @staticmethod
    def generate_prediction_for_one_outcome(
        unique_id: UUID,
        model: KnownModelName,
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
            func=lambda market_address: OmenSubgraphHandler().get_omen_market_by_market_id(
                market_id=market_address
            ),
        )
        return [CorrelatedMarketInput.from_omen_market(market) for market in markets]

    @observe()
    def generate_final_decision(
        self,
        question: str,
        scenarios_with_probabilities: list[t.Tuple[str, AnswerWithScenario]],
        created_time: DatetimeUTC | None,
        research_report: str | None = None,
    ) -> ProbabilisticAnswer:
        predictor = self._get_predictor(self.model)

        task_final_decision = Task(
            description=(
                FINAL_DECISION_PROMPT
                if not research_report
                else FINAL_DECISION_WITH_RESEARCH_PROMPT
            ),
            agent=predictor,
            expected_output=PROBABILITY_CLASS_OUTPUT,
            output_pydantic=ProbabilisticAnswer,
        )

        crew = Crew(agents=[predictor], tasks=[task_final_decision], verbose=True)

        correlated_markets = self.get_correlated_markets(question)

        event_date = get_event_date_from_question(question)
        n_remaining_days = (event_date - utcnow()).days if event_date else "Unknown"
        n_market_open_days = (
            (utcnow() - created_time).days if created_time else "Unknown"
        )
        logger.info(
            f"Event date is {event_date} and {n_remaining_days} days remaining. Market is already open for {n_market_open_days} days."
        )

        logger.info(
            f"Starting to generate final decision for '{question}' based on {len(scenarios_with_probabilities)=} and {len(correlated_markets)=}."
        )
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
        output = crew.kickoff(inputs=inputs)
        answer: ProbabilisticAnswer = output.pydantic
        answer_with_scenario = AnswerWithScenario.build_from_probabilistic_answer(
            answer, scenario=question, question=question
        )
        self.save_answer_to_long_term_memory(answer_with_scenario)
        logger.info(
            f"The final prediction has p_yes={answer.p_yes}, p_no={answer.p_no}, and confidence={answer.confidence}"
        )
        return answer

    @observe()
    def answer_binary_market(
        self,
        question: str,
        n_iterations: int = 1,
        created_time: DatetimeUTC | None = None,
    ) -> ProbabilisticAnswer | None:
        hypothetical_scenarios = self.get_hypohetical_scenarios(question)
        conditional_scenarios = self.get_required_conditions(question)

        unique_id = uuid4()
        observe_unique_id(unique_id)

        scenarios_with_probs: list[tuple[str, AnswerWithScenario]] = []
        for iteration in range(n_iterations):
            # If n_ierations is > 1, the agent will generate predictions for
            # each scenario multiple times, taking into account the previous
            # predictions. i.e. the probabilities are adjusted iteratively.
            logger.info(
                f"Starting to generate predictions for each scenario, iteration {iteration + 1} / {n_iterations}."
            )

            all_scenarios = (
                hypothetical_scenarios.scenarios + conditional_scenarios.scenarios
            )
            sub_predictions = par_generator(
                items=[
                    (
                        self.enable_langfuse,
                        unique_id,
                        self.model_for_generate_prediction_for_one_outcome,
                        scenario,
                        question,
                        scenarios_with_probs,
                        self.generate_prediction_for_one_outcome,
                    )
                    for scenario in all_scenarios
                ],
                func=process_scenario,
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

            if len(scenarios_with_probs) < len(all_scenarios) / 2:
                raise ValueError(
                    f"Too many of sub_predictions have failed, stopping the agent. Got only {len(scenarios_with_probs)} out of {len(all_scenarios)}."
                )

        final_answer = self.generate_final_decision(
            question, scenarios_with_probs, created_time=created_time
        )
        return final_answer


class ThinkThoroughlyWithItsOwnResearch(ThinkThoroughlyBase):
    identifier = THINK_THOROUGHLY
    model = "gpt-4-turbo-2024-04-09"
    model_for_generate_prediction_for_one_outcome = "gpt-4-turbo-2024-04-09"

    @staticmethod
    def generate_prediction_for_one_outcome(
        unique_id: UUID,
        model: KnownModelName,
        scenario: str,
        original_question: str,
        previous_scenarios_and_answers: (
            list[tuple[str, AnswerWithScenario]] | None
        ) = None,
    ) -> AnswerWithScenario | None:
        observe_unique_id(unique_id)

        researcher = ThinkThoroughlyWithItsOwnResearch._get_researcher(model)
        predictor = ThinkThoroughlyWithItsOwnResearch._get_predictor(model)

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
            output_pydantic=ProbabilisticAnswer,
            context=[task_research_one_outcome],
        )
        crew = Crew(
            agents=[researcher, predictor],
            tasks=[task_research_one_outcome, task_create_probability_for_one_outcome],
            verbose=True,
            process=Process.sequential,
        )

        inputs = {"sentence": scenario}
        if previous_scenarios_and_answers:
            inputs["previous_scenarios_with_probabilities"] = "\n".join(
                f"- Scenario '{s}' has probability of happening {a.p_yes * 100:.2f}% with confidence {a.confidence * 100:.2f}%, because {a.reasoning}"
                for s, a in previous_scenarios_and_answers
            )

        output = crew.kickoff(inputs=inputs)
        answer: ProbabilisticAnswer = output.pydantic

        if (
            task_research_one_outcome.tools_errors > 0
            or task_create_probability_for_one_outcome.tools_errors > 0
        ):
            logger.warning(
                f"Could not retrieve reasonable prediction for '{scenario}' because of errors in the tools"
            )
            return None

        answer_with_scenario = AnswerWithScenario.build_from_probabilistic_answer(
            answer, scenario=scenario, question=original_question
        )
        return answer_with_scenario


class ThinkThoroughlyWithPredictionProphetResearch(ThinkThoroughlyBase):
    identifier = THINK_THOROUGHLY_PROPHET
    model = "gpt-4-turbo-2024-04-09"
    model_for_generate_prediction_for_one_outcome = "gpt-4o-2024-08-06"

    @staticmethod
    def generate_prediction_for_one_outcome(
        unique_id: UUID,
        model: KnownModelName,
        scenario: str,
        original_question: str,
        previous_scenarios_and_answers: (
            list[tuple[str, AnswerWithScenario]] | None
        ) = None,
    ) -> AnswerWithScenario | None:
        observe_unique_id(unique_id)

        if previous_scenarios_and_answers:
            raise ValueError(
                "This agent does not support generating predictions with previous scenarios and answers in mind."
            )

        api_keys = APIKeys()

        research = prophet_research(
            goal=scenario,
            initial_subqueries_limit=0,  # This agent is making his own subqueries, so we don't need to generate another ones in the research part.
            max_results_per_search=5,
            min_scraped_sites=2,
            agent=PydanticAIAgent(
                OpenAIModel(
                    model,
                    provider=get_openai_provider(api_keys.openai_api_key),
                ),
                model_settings=ModelSettings(temperature=0.7),
            ),
            openai_api_key=api_keys.openai_api_key,
            tavily_api_key=api_keys.tavily_api_key,
        )
        prediction = prophet_make_prediction(
            market_question=scenario,
            additional_information=research.report,
            agent=PydanticAIAgent(
                OpenAIModel(
                    model,
                    provider=get_openai_provider(api_keys.openai_api_key),
                ),
                model_settings=ModelSettings(temperature=LLM_SUPER_LOW_TEMPERATURE),
            ),
            include_reasoning=True,
        )

        if prediction is None:
            logger.warning(
                f"ThinkThoroughlyWithPredictionProhpetResearch didn't generate prediction for '{scenario}'."
            )
            return None

        return AnswerWithScenario(
            scenario=scenario,
            original_question=original_question,
            p_yes=prediction.p_yes,
            confidence=prediction.confidence,
            reasoning=prediction.reasoning,
        )

    def generate_final_decision(
        self,
        question: str,
        scenarios_with_probabilities: list[t.Tuple[str, AnswerWithScenario]],
        created_time: DatetimeUTC | None,
        research_report: str | None = None,
    ) -> ProbabilisticAnswer:
        api_keys = APIKeys()
        report = (
            research_report
            or prophet_research(
                goal=question,
                agent=PydanticAIAgent(
                    OpenAIModel(
                        self.model,
                        provider=get_openai_provider(api_keys.openai_api_key),
                    ),
                    model_settings=ModelSettings(temperature=0.7),
                ),
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


def observe_unique_id(unique_id: UUID) -> None:
    # Used to mark the parent procses and its children with the same unique_id, so that we can link them together in Langfuse.
    langfuse_context.update_current_observation(
        metadata={
            "unique_id": str(unique_id),
        }
    )


def process_scenario(
    inputs: tuple[
        bool,
        UUID,
        KnownModelName,
        str,
        str,
        list[tuple[str, AnswerWithScenario]] | None,
        t.Callable[
            [
                UUID,
                KnownModelName,
                str,
                str,
                list[tuple[str, AnswerWithScenario]] | None,
            ],
            AnswerWithScenario | None,
        ],
    ],
) -> tuple[str, AnswerWithScenario | None]:
    # Needs to be a normal function outside of class, because `lambda` and `self` aren't pickable for processpool executor,
    # and process pool executor is required, because ChromaDB isn't thread-safe.
    # Input arguments needs to be as a single tuple, because par_generator requires a single argument.
    (
        enable_langfuse,
        unique_id,
        model,
        scenario,
        original_question,
        scenarios_with_probs,
        process_function,
    ) = inputs
    # Reset Langfuse, as this is executed as a separate process and Langfuse isn't thread-safe.
    initialize_langfuse(enable_langfuse)
    # Same for patching logger. Force patch, because while our logger is forked patched, LiteLLM still needs patching.
    patch_logger(force_patch=True)
    try:
        result = observe(name="process_scenario")(process_function)(
            unique_id, model, scenario, original_question, scenarios_with_probs
        )
    except Exception as e:
        # Log only as warning, because ThinkThoroughly is generating a lot of scenarios and it can happen that some of them will fail for any random error.
        # If too many of them fail, it will be logged as error and we will know.
        logger.warning(f"Error in `process_scenario` subprocess: {str(e)}")
        result = None
    return (scenario, result)
