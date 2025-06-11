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
from prediction_market_agent_tooling.deploy.betting_strategy import (
    MultiCategoricalMaxAccuracyBettingStrategy,
)
from prediction_market_agent_tooling.gtypes import USD
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


def test_single_agent(
    dataset_path: str,
    agent_name: str,
    max_trades_to_test_on: int = 3000,
    delay_between_trades: float = 2.0,
    max_bet_amount_min: USD = USD(1),
    max_bet_amount_max: USD = USD(5),
    trading_balance: USD = USD(40),
    max_price_impact: float = 0.7,
    use_old_research: bool = False,
    use_old_prediction: bool = False,
) -> None:
    dataset = pd.read_csv(dataset_path)

    # Filter dataset for the specific agent
    agent_data = dataset[dataset["agent_name"] == agent_name]
    if agent_data.empty:
        available_agents = dataset["agent_name"].unique().tolist()
        logger.error(
            f"Agent '{agent_name}' not found in dataset. Available agents: {available_agents}"
        )
        raise ValueError(f"Agent '{agent_name}' not found in dataset")

    logger.info(f"Testing agent: {agent_name} with {len(agent_data)} trades available")

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

    strategy = MultiCategoricalMaxAccuracyBettingStrategy(bet_amount=USD(10))
    tester = ProphetAgentTester(
        prophet_research=execute_prophet_research(),
        prophet_predict=execute_prophet_predict(),
        betting_strategy=strategy,
        use_old_research=use_old_research,
        use_old_prediction=use_old_prediction,
        max_trades_to_test_on=max_trades_to_test_on,
        run_name=f"test_{agent_name}",
        mocked_agent_name=agent_name,
        delay_between_trades=delay_between_trades,
    )

    test_results = tester.test_prophet_agent(
        agent_data, research_agent, prediction_agent
    )
    tester.evaluate_results(test_results, print_individual_metrics=True)

    trades_processed = len(test_results)
    logger.info(
        f"Completed testing for {agent_name}: {trades_processed} trades processed"
    )


@app.command()
def main(
    dataset_path: str = typer.Argument(..., help="Path to the CSV dataset file"),
    agent_name: str = typer.Argument(..., help="Name of the agent to test"),
    max_trades: int = typer.Option(
        3000, "--max-trades", help="Maximum number of trades to test per agent"
    ),
    delay: float = typer.Option(
        2.0, "--delay", help="Delay in seconds between processing each trade"
    ),
    min_bet: float = typer.Option(1.0, "--min-bet", help="Minimum bet amount"),
    max_bet: float = typer.Option(5.0, "--max-bet", help="Maximum bet amount"),
    balance: float = typer.Option(40.0, "--balance", help="Total trading balance"),
    max_impact: float = typer.Option(0.7, "--max-impact", help="Maximum price impact"),
    use_old_research: bool = typer.Option(
        True, "--use-old-research", help="Use old research generation"
    ),
    use_old_prediction: bool = typer.Option(
        True, "--use-old-prediction", help="Use old prediction generation"
    ),
) -> None:
    logger.info(f"Starting agent testing with dataset: {dataset_path}")

    test_single_agent(
        dataset_path=dataset_path,
        agent_name=agent_name,
        max_trades_to_test_on=max_trades,
        delay_between_trades=delay,
        max_bet_amount_min=USD(min_bet),
        max_bet_amount_max=USD(max_bet),
        trading_balance=USD(balance),
        max_price_impact=max_impact,
        use_old_research=use_old_research,
        use_old_prediction=use_old_prediction,
    )

    logger.info("Testing completed successfully!")


if __name__ == "__main__":
    app()
