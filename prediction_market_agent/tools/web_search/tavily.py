import tenacity
from prediction_market_agent_tooling.tools.cache import persistent_inmemory_cache
from prediction_market_agent_tooling.tools.langfuse_ import observe
from pydantic import BaseModel
from tavily import TavilyClient

from prediction_market_agent.utils import APIKeys


class WebSearchResult(BaseModel):
    url: str
    query: str


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(1), reraise=True
)
@persistent_inmemory_cache
@observe()
def web_search(query: str, max_results: int) -> list[WebSearchResult]:
    """
    Web search using Tavily API.
    """
    tavily = TavilyClient(api_key=APIKeys().tavily_api_key.get_secret_value())
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
