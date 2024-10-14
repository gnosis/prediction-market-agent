from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.tavily_storage.tavily_models import (
    TavilyStorage,
)
from prediction_prophet.benchmark.agents import (  # noqa: F401 # Just to make it available for the user of research.
    _make_prediction as prophet_make_prediction,
)
from prediction_prophet.functions.research import Research
from prediction_prophet.functions.research import research as original_research
from pydantic.types import SecretStr


def prophet_research(
    goal: str,
    model: str,
    openai_api_key: SecretStr,
    tavily_api_key: SecretStr,
    tavily_storage: TavilyStorage | None,
    initial_subqueries_limit: int = 20,
    subqueries_limit: int = 4,
    max_results_per_search: int = 5,
    min_scraped_sites: int = 10,
) -> Research:
    """
    Use `min_scraped_sites` as a proxy for setting a minimum requirement for
    'how thorough the research must be'. Up to (subqueries_limit * max_results_per_search)
    sites will be scraped, but the actual number may be less because of:

    - duplication of URLs across searches
    - failed scrapes
    - insufficient search results (although unlikely)

    If the number of scraped sites is less than `min_scraped_sites`, an error
    will be raised.
    """
    return original_research(
        goal=goal,
        model=model,
        use_summaries=False,
        initial_subqueries_limit=initial_subqueries_limit,
        subqueries_limit=subqueries_limit,
        max_results_per_search=max_results_per_search,
        min_scraped_sites=min_scraped_sites,
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        tavily_storage=tavily_storage,
        logger=logger,
    )
