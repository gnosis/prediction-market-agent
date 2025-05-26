from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.logprobs_parser import (
    FieldLogprobs,
    LogprobsParser,
)
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.tools.perplexity.perplexity_models import (
    PerplexityResponse,
)
from prediction_market_agent_tooling.tools.perplexity.perplexity_search import (
    perplexity_search,
)
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.settings import ModelSettings

from prediction_market_agent.agents.logprobs_oai_model import LogProbsOpenAIModel
from prediction_market_agent.tools.openai_utils import get_openai_provider
from prediction_market_agent.tools.web_scrape.structured_summary import (
    web_scrape_structured_and_summarized,
)
from prediction_market_agent.utils import APIKeys


class DecidabilityResponse(BaseModel):
    rationale: str
    answer: Literal["YES", "NO"]


class PredictionResponse(BaseModel):
    rationale: str
    p_yes: float = Field(ge=0, le=1)


PERPLEXITY_QUERY = """
Your goal is to provide assesment of sources for following question: {market_question}?

* Find relevant information. Every time evaluate quality and relevance of your soruces.
* Don't answer the question, just provide assesment of quality and relevance of ALL given sources.
* Each assesment MUST start with LINK OF THE SOURCE so i can pair it with given source later.
* In your assesment also provide critique of given source - website etc..
* Make sure that assesment is returned in good readble format, separated by bullet points and paragraphs for each source.
"""

SUMMARY_OBJECTIVE = (
    "Summarize all relevant information regarding the question: {market_question}."
)


DECIDABILITY_QUERRY = """
Given the question: {market_question}, critique of weak critique of sources: {weak_critique} and summaries of sources: {summary} decide if the question is solvable.
* Your goal is not to solve the question. You need only to asses if the question is solvable.
* It is solvable if it is possible to make prediction based on given sources, when taking cirtique in to the account.
* You dont care about the prediction you only look on the sources and information in it.
* Retrurn response in following format:
* First return your rationale of why did you decided given way after field "rationale": "YOUR RATIONALE HERE",\n
* Then return your answer, YES or NO in field "answer": "YOUR RESPONSE HERE"\n

* RETURN ONLY RESPONSE CONTAINING ONLY FOLLOWING NOTHING ELSE  "rationale": "YOUR RATIONALE HERE",\n "answer": "YOUR RESPONSE HERE",\n
"""

PREDICTION_QUERRY = """
Given the question: {market_question}, critique of sources: {critique} and summaries of sources: {summary} predict the outcome of the question.

* Evaluate all relevant sources and provided information.
* Evaluate critique of the sources to ignore ones that are not relevant or useless.
* Predict positive outcome of the question in the filed "p_yes" in range [0, 1]

* Return response in following format:
* First return your rationale of why did you decided given way in "rationale": "YOUR RATIONALE HERE",\n
* Then return your positive prediction percentate inside of field "p_yes": "YOUR PREDICTION  IN RANGE [0-1]"\n

* Your output response must be only a single JSON object to be parsed by Python's "json.loads()".
* RETURN ONLY RESPONSE CONTAINING ONLY FOLLOWING NOTHING ELSE  "rationale": "YOUR RATIONALE HERE",\n "p_yes": "YOUR PREDICTION  IN RANGE [0-1]",\n
* Output only the JSON object in your response. Do not include any other contents in your response.

"""


class DeployableLogProbsAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 4

    def load(
        self,
        use_solvability_score: bool = False,
        min_solvability_score: float = 0.3,
    ) -> None:
        super().load()

        self.research_agent = Agent(
            LogProbsOpenAIModel(
                "gpt-4o", provider=get_openai_provider(api_key=APIKeys().openai_api_key)
            ),
            model_settings=ModelSettings(
                temperature=0, extra_body={"logprobs": True, "top_logprobs": 3}
            ),
        )
        self.logprobs_parser = LogprobsParser()
        self.api_key = APIKeys()
        # if solvability score is used,
        self.use_solvability_score = use_solvability_score
        self.min_solvability_score = min_solvability_score

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        logger.info(f"Answering market: {market.question}")

        research_results = self._do_perplexity_search(market.question)
        logger.info(f"Research results: {research_results.content}")
        logger.info(
            f"Research results citations for links: {research_results.citations}"
        )

        summaries = self._process_links_parallel(
            market.question, research_results.citations, max_workers=10
        )
        logger.info(f"Summaries of sources: {summaries}")

        if self.use_solvability_score:
            solvability_score = self._calculate_solvability_score(
                market.question, research_results.content, summaries
            )
            logger.info(f"Solvability score: {solvability_score}")
            if (
                solvability_score is not None
                and solvability_score < self.min_solvability_score
            ):
                logger.info(
                    f"Skipping market, min solvability {self.min_solvability_score} not reached"
                )
                return None

        prediction_result = self._predict_market_outcome(
            market.question, research_results.content, summaries
        )
        logger.info(f"Predicted market outcome: {prediction_result}")
        return prediction_result

    def _do_perplexity_search(self, market_question: str) -> PerplexityResponse:
        return perplexity_search(
            query=PERPLEXITY_QUERY.format(market_question=market_question),
            api_keys=self.api_key,
        )

    def _process_links_parallel(
        self, market_question: str, links: list[str], max_workers: int = 10
    ) -> list[str]:
        summaries: list[str] = []

        def process_single_link(link: str) -> str | None:
            try:
                return (
                    link
                    + " \n "
                    + web_scrape_structured_and_summarized(
                        SUMMARY_OBJECTIVE.format(market_question=market_question),
                        link,
                        remove_a_links=True,
                    )
                )
            except Exception as e:
                logger.warning(f"Error processing link {link}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single_link, links))

        return [result for result in results if result is not None]

    def _calculate_solvability_score(
        self, market_question: str, critique: str, crawl_summaries: list[str]
    ) -> float | None:
        solvability_response = self.research_agent.run_sync(
            DECIDABILITY_QUERRY.format(
                market_question=market_question,
                weak_critique=critique,
                summary=crawl_summaries,
            )
        )

        raw_logprobs = self.get_logprobs(solvability_response)
        if raw_logprobs is None:
            raise ValueError("No logprobs found in solvability response")

        # Get probabilities for "YES" and "NO" tokens in the answer field
        probs, _ = self._extract_token_probabilities(
            raw_logprobs=raw_logprobs,
            model_type=DecidabilityResponse,
            field_key="answer",
            expected_tokens=["YES", "NO"],
        )

        yes_prob = probs.get("YES", None)
        no_prob = probs.get("NO", None)

        if yes_prob and no_prob and yes_prob > no_prob:
            return yes_prob
        elif no_prob and no_prob > 0:
            return 1 - no_prob

        return None

    def _extract_token_probabilities(
        self,
        raw_logprobs: list[dict[str, Any]],
        model_type: type,
        field_key: str,
        expected_tokens: list[str] | None = None,
    ) -> tuple[dict[str, float], list[FieldLogprobs]]:
        raw_field_logprobs = self.logprobs_parser.parse_logprobs(
            raw_logprobs, model_type
        )
        field_logprobs = [
            logprob for logprob in raw_field_logprobs if logprob.key == field_key
        ]

        top_logprobs = (
            field_logprobs[0].logprobs
            if field_logprobs and len(field_logprobs) > 0
            else []
        )
        if not top_logprobs:
            logger.warning(f"No logprobs found for field {field_key}")

        return {
            logprob.token.upper(): logprob.prob
            for logprob in top_logprobs
            if not expected_tokens or logprob.token.upper() in expected_tokens
        }, raw_field_logprobs

    def get_logprobs(self, result: AgentRunResult) -> list[dict[str, Any]] | None:
        logprobs = None
        messages = result.all_messages()
        if messages and hasattr(messages[-1], "vendor_details"):
            vendor_details = messages[-1].vendor_details
            if vendor_details:
                logprobs = vendor_details.get("logprobs", None)

        return logprobs

    def _predict_market_outcome(
        self, market_question: str, critique: str, crawl_summaries: list[str]
    ) -> ProbabilisticAnswer | None:
        prediction = self.research_agent.run_sync(
            PREDICTION_QUERRY.format(
                market_question=market_question,
                critique=critique,
                summary=crawl_summaries,
            )
        )
        raw_logprobs = self.get_logprobs(prediction)
        if raw_logprobs is None:
            raise ValueError("No logprobs found in prediction")

        confidence, raw_field_logprobs = self._extract_token_probabilities(
            raw_logprobs=raw_logprobs,
            model_type=PredictionResponse,
            field_key="p_yes",
        )

        if not confidence:
            raise ValueError("No confidence scores found in logprobs")

        max_confidence_key: str = max(confidence.items(), key=lambda item: item[1])[0]
        response = clean_json_response(prediction.data)

        return ProbabilisticAnswer(
            p_yes=Probability(round(float(response.p_yes), 2)),
            confidence=round(confidence[max_confidence_key], 2),
            reasoning=response.rationale,
            logprobs=raw_field_logprobs,
        )


def clean_json_response(response_str: str) -> PredictionResponse:
    response_str = response_str.strip().replace("\n", " ")
    response_str = " ".join(response_str.split())
    start_index = response_str.find("{")
    end_index = response_str.rfind("}")
    response_str = response_str[start_index : end_index + 1]

    return PredictionResponse.model_validate_json(response_str)
