from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_prophet.benchmark.agents import Prediction
from prediction_prophet.benchmark.agents import (
    _make_prediction as prophet_make_prediction,
)
from prediction_prophet.functions.research import Research, research
from pydantic.types import SecretStr


def prophet_research(
    goal: str,
    model: str,
    openai_api_key: SecretStr,
    tavily_api_key: SecretStr,
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 4,
    max_results_per_search: int = 5,
    min_scraped_sites: int = 10,
    add_langfuse_callback: bool = False,
) -> Research:
    """
    Use `min_scraped_sites` as a proxy for setting a minimum requirement for
    'how thorough the research must be'. Up to (min_scraped_sites * max_results_per_search)
    sites will be scraped, but the actual number may be less because of:

    - duplication of URLs across searches
    - failed scrapes
    - insufficient search results (although unlikely)

    If the number of scraped sites is less than `min_scraped_sites`, an error
    will be raised.
    """
    return research(
        goal=goal,
        model=model,
        use_summaries=False,
        initial_subqueries_limit=initial_subqueries_limit,
        subqueries_limit=subqueries_limit,
        max_results_per_search=max_results_per_search,
        min_scraped_sites=min_scraped_sites,
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        logger=logger,
        add_langfuse_callback=add_langfuse_callback,
    )


@observe()
def prophet_research_observed(
    goal: str,
    model: str,
    openai_api_key: SecretStr,
    tavily_api_key: SecretStr,
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 4,
    max_results_per_search: int = 5,
    min_scraped_sites: int = 10,
) -> Research:
    return prophet_research(
        goal=goal,
        model=model,
        initial_subqueries_limit=initial_subqueries_limit,
        subqueries_limit=subqueries_limit,
        max_results_per_search=max_results_per_search,
        min_scraped_sites=min_scraped_sites,
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        add_langfuse_callback=True,
    )


@observe()
def prophet_make_prediction_observed(
    market_question: str,
    additional_information: str,
    engine: str,
    temperature: float,
    api_key: SecretStr | None = None,
) -> Prediction:
    return prophet_make_prediction(
        market_question=market_question,
        additional_information=additional_information,
        engine=engine,
        temperature=temperature,
        api_key=api_key,
        add_langfuse_callback=True,
    )
