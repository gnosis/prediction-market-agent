import tenacity
from prediction_market_agent_tooling.tools.cache import persistent_inmemory_cache
from prediction_market_agent_tooling.tools.utils import secret_str_from_env
from pydantic import BaseModel
from tavily import TavilyClient


class WebSearchResult(BaseModel):
    url: str
    query: str


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1), reraise=True
)
@persistent_inmemory_cache
def web_search(query: str, max_results: int) -> list[WebSearchResult]:
    """
    Web search using Tavily API.
    """
    tavily_api_key = secret_str_from_env("TAVILY_API_KEY")
    tavily = TavilyClient(api_key=tavily_api_key.get_secret_value())
    response = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_raw_content=True,
    )

    results = [
        WebSearchResult(
            url=result["url"],
            query=query,
        )
        for result in response["results"]
    ]

    return results
