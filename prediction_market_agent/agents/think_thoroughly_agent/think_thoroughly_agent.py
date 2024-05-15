import datetime
import sys
import typing as t

from crewai import Agent, Crew, Process, Task
from langchain.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.pydantic_v1 import SecretStr
from langchain_openai import ChatOpenAI
from openai import APIError
from prediction_market_agent_tooling.deploy.agent import Answer
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import FilterBy, SortBy
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.parallelism import par_generator
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic import BaseModel

from prediction_market_agent.agents.think_thoroughly_agent.models import (
    CorrelatedMarketInput,
)
from prediction_market_agent.agents.think_thoroughly_agent.pinecone_utils import (
    create_metadatas_from_omen_markets,
    create_texts_from_omen_markets,
)
from prediction_market_agent.agents.think_thoroughly_agent.prompts import (
    CREATE_HYPOTHETICAL_SCENARIOS_FROM_SCENARIO_PROMPT,
    CREATE_REQUIRED_CONDITIONS_PROMPT,
    FINAL_DECISION_PROMPT,
    LIST_OF_SCENARIOS_OUTPUT,
    PROBABILITY_CLASS_OUTPUT,
    PROBABILITY_FOR_ONE_OUTCOME_PROMPT,
    RESEARCH_OUTCOME_OUTPUT,
    RESEARCH_OUTCOME_PROMPT,
    RESEARCH_OUTCOME_WITH_PREVIOUS_OUTPUTS_PROMPT,
)
from prediction_market_agent.db.pinecone_handler import PineconeHandler
from prediction_market_agent.utils import APIKeys


class Scenarios(BaseModel):
    scenarios: list[str]


class CrewAIAgentSubquestions:
    pinecone_handler: t.Optional[PineconeHandler] = None
    model: str

    def __init__(self, model: str) -> None:
        self.model = model
        self.subgraph_handler = OmenSubgraphHandler()

    def _init_pinecone(self) -> None:
        if not self.pinecone_handler:
            self.pinecone_handler = PineconeHandler()

    def _get_current_date(self) -> str:
        return utcnow().strftime("%Y-%m-%d")

    def _get_researcher(self) -> Agent:
        return Agent(
            role="Research Analyst",
            goal="Research and report on some future event, giving high quality and nuanced analysis",
            backstory=f"Current date is {self._get_current_date()}. You are a senior research analyst who is adept at researching and reporting on future events.",
            verbose=True,
            allow_delegation=False,
            tools=[self._build_tavily_search()],
            llm=self._build_llm(),
        )

    def _get_predictor(self) -> Agent:
        return Agent(
            role="Professional Gambler",
            goal="Predict, based on some research you are presented with, whether or not a given event will occur",
            backstory=f"Current date is {self._get_current_date()}. You are a professional gambler who is adept at predicting and betting on the outcomes of future events.",
            verbose=True,
            allow_delegation=False,
            llm=self._build_llm(),
        )

    def _build_tavily_search(self) -> TavilySearchResults:
        api_key = SecretStr(APIKeys().tavily_api_key.get_secret_value())
        api_wrapper = TavilySearchAPIWrapper(tavily_api_key=api_key)
        return TavilySearchResults(api_wrapper=api_wrapper)

    def _build_llm(self) -> BaseChatModel:
        keys = APIKeys()
        # ToDo - Add Langfuse callback handler here once integration becomes clear (see
        #  https://github.com/gnosis/prediction-market-agent/issues/107)
        llm = ChatOpenAI(
            model=self.model,
            api_key=keys.openai_api_key.get_secret_value(),
            temperature=0.0,
        )
        return llm

    def update_markets(self) -> None:
        """We use the agent's run to add embeddings of new markets that don't exist yet in the
        vector DB."""
        created_after = utcnow() - datetime.timedelta(days=7)
        markets = self.subgraph_handler.get_omen_binary_markets_simple(
            limit=sys.maxsize,
            filter_by=FilterBy.OPEN,
            sort_by=SortBy.NEWEST,
            created_after=created_after,
        )
        texts = create_texts_from_omen_markets(markets)
        metadatas = create_metadatas_from_omen_markets(markets)
        if texts:
            logger.debug(f"Inserting {len(texts)} into the vector database.")
            self._init_pinecone()
            self.pinecone_handler.insert_texts_if_not_exists(texts=texts, metadatas=metadatas)  # type: ignore[union-attr]

    def get_required_conditions(self, question: str) -> Scenarios:
        researcher = self._get_researcher()

        create_required_conditions = Task(
            description=CREATE_REQUIRED_CONDITIONS_PROMPT,
            expected_output=LIST_OF_SCENARIOS_OUTPUT,
            output_json=Scenarios,
            agent=researcher,
        )

        report_crew = Crew(
            agents=[researcher],
            tasks=[create_required_conditions],
        )
        result = report_crew.kickoff(inputs={"scenario": question, "n_scenarios": 3})
        scenarios = Scenarios.model_validate_json(result)

        logger.info(f"Created conditional scenarios: {scenarios.scenarios}")
        return scenarios

    def get_hypohetical_scenarios(self, question: str) -> Scenarios:
        researcher = self._get_researcher()

        create_scenarios_task = Task(
            description=CREATE_HYPOTHETICAL_SCENARIOS_FROM_SCENARIO_PROMPT,
            expected_output=LIST_OF_SCENARIOS_OUTPUT,
            output_json=Scenarios,
            agent=researcher,
        )

        report_crew = Crew(
            agents=[researcher],
            tasks=[create_scenarios_task],
        )
        result = report_crew.kickoff(inputs={"scenario": question, "n_scenarios": 5})
        scenarios = Scenarios.model_validate_json(result)

        # Add the original question if it wasn't included by the LLM.
        if question not in scenarios.scenarios:
            scenarios.scenarios.append(question)

        logger.info(f"Created possible hypothetical scenarios: {scenarios.scenarios}")
        return scenarios

    def generate_prediction_for_one_outcome(
        self,
        sentence: str,
        previous_scenarios_and_answers: list[tuple[str, Answer]] | None = None,
    ) -> Answer | None:
        researcher = self._get_researcher()
        predictor = self._get_predictor()

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
            output_json=Answer,
            context=[task_research_one_outcome],
        )
        crew = Crew(
            agents=[researcher, predictor],
            tasks=[task_research_one_outcome, task_create_probability_for_one_outcome],
            verbose=2,
            process=Process.sequential,
        )

        inputs = {"sentence": sentence}
        if previous_scenarios_and_answers:
            inputs["previous_scenarios_with_probabilities"] = "\n".join(
                f"- Scenario '{s}' has probability of happening {a.p_yes * 100:.2f}% with confidence {a.confidence * 100:.2f}%, because {a.reasoning}"
                for s, a in previous_scenarios_and_answers
            )

        try:
            result = crew.kickoff(inputs=inputs)
        except APIError as e:
            logger.error(
                f"Could not retrieve response from the model provider because of {e}"
            )
            return None

        try:
            output = Answer.model_validate_json(result)
            return output
        except ValueError as e:
            logger.error(
                f"Could not parse the result ('{result}') as Answer because of {e}"
            )
            return None

    def get_correlated_markets(self, question: str) -> list[CorrelatedMarketInput]:
        self._init_pinecone()
        nearest_questions = self.pinecone_handler.find_nearest_questions(5, text=question)  # type: ignore[union-attr]

        markets = list(
            par_generator(
                [q.market_address for q in nearest_questions],
                lambda market_address: self.subgraph_handler.get_omen_market_by_market_id(
                    market_id=market_address
                ),
            )
        )
        return [CorrelatedMarketInput.from_omen_market(market) for market in markets]

    def generate_final_decision(
        self, question: str, scenarios_with_probabilities: list[t.Tuple[str, Answer]]
    ) -> Answer:
        predictor = self._get_predictor()

        task_final_decision = Task(
            description=FINAL_DECISION_PROMPT,
            agent=predictor,
            expected_output=PROBABILITY_CLASS_OUTPUT,
            output_json=Answer,
        )

        crew = Crew(
            agents=[predictor],
            tasks=[task_final_decision],
            verbose=2,
        )

        correlated_markets = self.get_correlated_markets(question)

        logger.info(f"Starting to generate final decision for '{question}'.")
        crew.kickoff(
            inputs={
                "scenarios_with_probabilities": "\n".join(
                    f"- Scenario '{s}' has probability of happening {a.p_yes * 100:.2f}% with confidence {a.confidence * 100:.2f}%, because {a.reasoning}"
                    for s, a in scenarios_with_probabilities
                ),
                "number_of_scenarios": len(scenarios_with_probabilities),
                "scenario_to_assess": question,
                "correlated_markets": [r.dict() for r in correlated_markets],
            }
        )
        output = Answer.model_validate_json(task_final_decision.output.raw_output)
        logger.info(
            f"The final prediction is '{output.decision}', with p_yes={output.p_yes}, p_no={output.p_no}, and confidence={output.confidence}"
        )
        return output

    def answer_binary_market(
        self, question: str, n_iterations: int = 1
    ) -> Answer | None:
        hypothetical_scenarios = self.get_hypohetical_scenarios(question)
        conditional_scenarios = self.get_required_conditions(question)

        scenarios_with_probs: list[tuple[str, Answer]] = []
        for iteration in range(n_iterations):
            logger.info(
                f"Starting to generate predictions for each scenario, iteration {iteration + 1} / {n_iterations}."
            )

            sub_predictions = par_generator(
                hypothetical_scenarios.scenarios + conditional_scenarios.scenarios,
                lambda x: (
                    x,
                    self.generate_prediction_for_one_outcome(x, scenarios_with_probs),
                ),
            )

            scenarios_with_probs = []
            for scenario, prediction in sub_predictions:
                if prediction is None:
                    continue
                scenarios_with_probs.append((scenario, prediction))
                logger.info(
                    f"'{scenario}' has prediction {prediction.p_yes * 100:.2f}% chance of being True, because: '{prediction.reasoning}'"
                )

        final_answer = (
            self.generate_final_decision(question, scenarios_with_probs)
            if scenarios_with_probs
            else None
        )
        return final_answer
