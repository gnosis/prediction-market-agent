from functools import partial

import pandas as pd
import typer
from prediction_market_agent.tools.openai_utils import get_openai_provider
from prediction_market_agent.utils import APIKeys
from prediction_prophet.autonolas.research import Prediction as PredictionProphet
from prediction_prophet.autonolas.research import make_prediction
from prediction_prophet.functions.research import Research, research
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from prediction_market_agent_tooling.agent.development_tools.prophet_agent_tester import (
    ProphetAgentTester,
)
from prediction_market_agent_tooling.loggers import logger

app = typer.Typer()

GPT_4O_MODEL = "gpt-4o-2024-08-06"


def execute_prophet_research(
    use_summaries: bool = False,
    use_tavily_raw_content: bool = False,
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 5,
    max_results_per_search: int = 5,
    min_scraped_sites: int = 5,
) -> partial[Research]:
    return partial(
        research,
        use_summaries=use_summaries,
        use_tavily_raw_content=use_tavily_raw_content,
        initial_subqueries_limit=initial_subqueries_limit,
        subqueries_limit=subqueries_limit,
        max_results_per_search=max_results_per_search,
        min_scraped_sites=min_scraped_sites,
        logger=logger,
    )


def execute_prophet_predict(
    include_reasoning: bool = False,
) -> partial[PredictionProphet]:
    return partial(make_prediction, include_reasoning=include_reasoning)


def test_all_models(
    dataset_path: str,
    max_trades_to_test_on: int = 3000,
    use_old_research: bool = True,
    use_old_prediction: bool = True,
) -> None:
    dataset = pd.read_csv(dataset_path)

    agent_names = dataset["agent_name"].unique().tolist()
    total_agents = len(agent_names)
    logger.info(f"Found {total_agents} agents in dataset: {agent_names}")

    api_keys = APIKeys()
    research_agent = Agent(
        OpenAIModel(
            GPT_4O_MODEL,
            provider=get_openai_provider(api_key=api_keys.openai_api_key),
        ),
        model_settings=ModelSettings(temperature=0.7),
    )
    prediction_agent = Agent(
        OpenAIModel(
            GPT_4O_MODEL,
            provider=get_openai_provider(api_key=api_keys.openai_api_key),
        ),
        model_settings=ModelSettings(temperature=0.0),
    )

    all_results, all_metrics = {}, {}
    for agent_index, agent_name in enumerate(agent_names, 1):
        logger.info(f"Testing agent {agent_index}/{total_agents}: {agent_name}")

        tester = ProphetAgentTester(
            prophet_research=execute_prophet_research(),
            prophet_predict=execute_prophet_predict(),
            use_old_research=use_old_research,
            use_old_prediction=use_old_prediction,
            max_trades_to_test_on=max_trades_to_test_on,
            run_name=f"test_{agent_name}",
            mocked_agent_name=agent_name,
            simulate_trades=False,
        )

        test_results = tester.test_prophet_agent(
            dataset, research_agent, prediction_agent
        )
        evaluation_metrics = tester.evaluate_results(test_results)

        all_results[agent_name] = test_results
        all_metrics[agent_name] = evaluation_metrics

        trades_processed = len(test_results)
        logger.info(
            f"Completed testing for {agent_name}: {trades_processed} trades processed"
        )

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    total_trades_processed = 0
    for agent_index, agent_name in enumerate(agent_names, 1):
        metrics = all_metrics[agent_name]
        trades_count = len(all_results[agent_name])
        total_trades_processed += trades_count

        if metrics:
            logger.info(f"{agent_index}. {agent_name} ({trades_count} trades):")
            logger.info(
                f"   Binary Prediction Accuracy: {metrics.binary_prediction_accuracy:.4f}"
            )
            logger.info(
                f"   Weighted Prediction Accuracy: {metrics.weighted_prediction_accuracy:.4f}"
            )
            logger.info(f"   Brier Score: {metrics.prediction_brier_score:.4f}")
            if tester.simulate_trades:
                logger.info(
                    f"   Binary Trade Accuracy: {metrics.binary_trade_accuracy:.4f}"
                )
        else:
            logger.info(
                f"{agent_index}. {agent_name} ({trades_count} trades): No metrics available"
            )

    logger.info(
        f"\nTotal: {total_agents} agents tested, {total_trades_processed} total trades processed"
    )


@app.command()
def main(
    dataset_path: str = typer.Argument(..., help="Path to the CSV dataset file"),
    max_trades: int = typer.Option(
        3000, "--max-trades", help="Maximum number of trades to test per agent"
    ),
    use_old_research: bool = typer.Option(
        True, "--use-old-research", help="Use old research generation"
    ),
    use_old_prediction: bool = typer.Option(
        True, "--use-old-prediction", help="Use old prediction generation"
    ),
) -> None:
    logger.info(f"Starting agent testing with dataset: {dataset_path}")

    test_all_models(
        dataset_path=dataset_path,
        max_trades_to_test_on=max_trades,
        use_old_research=use_old_research,
        use_old_prediction=use_old_prediction,
    )

    logger.info("Testing completed successfully!")


if __name__ == "__main__":
    app()
