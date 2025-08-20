import json
from datetime import datetime, timedelta, timezone
from functools import partial

import pandas as pd
import typer
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.deploy.betting_strategy import (
    FullBinaryKellyBettingStrategy,
)
from prediction_market_agent_tooling.gtypes import USD
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_prophet.autonolas.research import (
    FIELDS_DESCRIPTIONS,
    PREDICTION_PROMPT,
    LogprobsParser,
)
from prediction_prophet.autonolas.research import Prediction
from prediction_prophet.autonolas.research import Prediction as PredictionProphet
from prediction_prophet.autonolas.research import (
    clean_completion_json,
    fields_dict_to_bullet_list,
    list_to_list_str,
)
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from prediction_market_agent.agents.logprobs_oai_model import LogProbsOpenAIModel
from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount
from prediction_market_agent.development_tools.prophet_agent_tester import (
    ProphetAgentTester,
)
from prediction_market_agent.development_tools.test_single_agent import (
    execute_prophet_research,
)
from prediction_market_agent.tools.openai_utils import get_openai_provider

app = typer.Typer()

GPT_4O_MODEL = "gpt-4o-2024-08-06"


def make_prediction_local(
    agent: Agent,
    prompt: str,
    additional_information: str,
    market: AgentMarket,
    include_reasoning: bool = True,
    use_logprobs: bool = False,
) -> Prediction:
    current_time_utc = (
        market.created_time
        if market.created_time
        else (datetime.now(timezone.utc) - timedelta(days=365))
    )
    current_time_utc = current_time_utc.replace(tzinfo=timezone.utc)
    formatted_time_utc = current_time_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-6] + "Z"

    field_descriptions = FIELDS_DESCRIPTIONS.copy()
    if use_logprobs:
        field_descriptions.pop("confidence")

    prediction_prompt = PREDICTION_PROMPT.format(
        user_prompt=prompt,
        additional_information=additional_information,
        n_fields=len(field_descriptions),
        fields_list=list_to_list_str(list(field_descriptions)),
        fields_description=fields_dict_to_bullet_list(field_descriptions),
        timestamp=formatted_time_utc,
    )
    result = agent.run_sync(prediction_prompt)

    completion = result.data

    logger.info(f"Completion: {completion}")
    completion_clean = clean_completion_json(completion)
    logger.info(f"Completion cleaned: {completion_clean}")

    response = json.loads(completion_clean)
    if use_logprobs:
        logprobs = None
        messages = result.all_messages()
        if messages and hasattr(messages[-1], "vendor_details"):
            vendor_details = messages[-1].vendor_details
            if vendor_details:
                logprobs = vendor_details.get("logprobs")

        if logprobs is None:
            raise ValueError("No logprobs found in vendor details")

        raw_field_logprobs = LogprobsParser(skip_fields=["reasoning"]).parse_logprobs(
            logprobs, Prediction
        )
        field_logprobs = [
            logprob for logprob in raw_field_logprobs if logprob.key == "p_yes"
        ]

        top_logprobs = (
            field_logprobs[0].logprobs
            if field_logprobs and len(field_logprobs) > 0
            else []
        )
        if not top_logprobs:
            raise ValueError("No logprobs found for field p_yes")

        p_yes_logprobs = {
            logprob.token.upper(): logprob.prob for logprob in top_logprobs
        }
        response["confidence"] = float(
            max(p_yes_logprobs.items(), key=lambda x: x[1])[1]
        )

    return Prediction.model_validate(response)


def execute_prophet_predict(
    include_reasoning: bool = True,
) -> partial[PredictionProphet]:
    return partial(
        make_prediction_local, include_reasoning=include_reasoning, use_logprobs=False
    )


def execute_prophet_predict_logprobs(
    include_reasoning: bool = True,
) -> partial[PredictionProphet]:
    return partial(
        make_prediction_local, include_reasoning=include_reasoning, use_logprobs=True
    )


def test_logprobs_agent(
    dataset_path: str,
    max_trades_to_test_on: int = 3000,
    delay_between_trades: float = 2.0,
    use_old_research: bool = True,
    use_old_prediction: bool = True,
) -> None:
    dataset = pd.read_csv(dataset_path)

    api_keys = APIKeys()
    research_agent = Agent(
        OpenAIModel(
            GPT_4O_MODEL,
            provider=get_openai_provider(api_key=api_keys.openai_api_key),
        ),
        model_settings=ModelSettings(temperature=0.7),
    )
    prediction_agent = Agent(
        LogProbsOpenAIModel(
            GPT_4O_MODEL,
            provider=get_openai_provider(api_key=api_keys.openai_api_key),
        ),
        model_settings=ModelSettings(
            temperature=0.0, extra_body={"logprobs": True, "top_logprobs": 3}
        ),
    )

    strategy = FullBinaryKellyBettingStrategy(
        max_position_amount=get_maximum_possible_bet_amount(
            min_=USD(1),
            max_=USD(5),
            trading_balance=USD(10),
        ),
        max_price_impact=0.7,
    )
    logprobs_tester = ProphetAgentTester(
        prophet_research=execute_prophet_research(),
        prophet_predict=execute_prophet_predict_logprobs(),
        betting_strategy=strategy,
        use_old_research=True,
        use_old_prediction=False,
        max_trades_to_test_on=max_trades_to_test_on,
        run_name="test_logprobs_gpt4o_agent",
        mocked_agent_name="DeployablePredictionProphetGPT4oAgent",
        delay_between_trades=delay_between_trades,
    )
    standard_4o_tester = ProphetAgentTester(
        prophet_research=execute_prophet_research(),
        prophet_predict=execute_prophet_predict(),
        betting_strategy=strategy,
        use_old_research=True,
        use_old_prediction=True,
        max_trades_to_test_on=max_trades_to_test_on,
        run_name="test_gpt4o_agent",
        mocked_agent_name="DeployablePredictionProphetGPT4oAgent",
        delay_between_trades=delay_between_trades,
    )

    logprobs_test_results, _ = logprobs_tester.test_prophet_agent(
        dataset, research_agent, prediction_agent
    )
    standard_4o_test_results, _ = standard_4o_tester.test_prophet_agent(
        dataset, research_agent, prediction_agent
    )

    logprobs_tester.evaluate_results(
        logprobs_test_results, print_individual_metrics=True
    )
    trades_processed = len(logprobs_test_results)
    logger.info(f"Completed testing with logprobs: {trades_processed} trades processed")

    standard_4o_tester.evaluate_results(
        standard_4o_test_results, print_individual_metrics=True
    )
    trades_processed = len(standard_4o_test_results)
    logger.info(f"Completed testing with 4o: {trades_processed} trades processed")


@app.command()
def main(
    dataset_path: str = typer.Argument(..., help="Path to the CSV dataset file"),
    max_trades: int = typer.Option(
        3000, "--max-trades", help="Maximum number of trades to test"
    ),
    delay: float = typer.Option(
        2.0, "--delay", help="Delay in seconds between processing each trade"
    ),
) -> None:
    logger.info(f"Starting logprobs agent testing with dataset: {dataset_path}")

    test_logprobs_agent(
        dataset_path=dataset_path,
        max_trades_to_test_on=max_trades,
        delay_between_trades=delay,
    )


if __name__ == "__main__":
    app()
