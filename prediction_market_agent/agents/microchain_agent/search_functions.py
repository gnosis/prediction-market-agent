from microchain import Function
from prediction_market_agent_tooling.tools.tavily.tavily_search import tavily_search


class TavilySearch(Function):
    @property
    def description(self) -> str:
        return "Use this function to do a Google search using Tavily search engine."

    @property
    def example_args(self) -> list[str]:
        return ["Your query to search for"]

    def __call__(self, query: str) -> str:
        response = tavily_search(query=query, search_depth="basic")
        results_as_text = "\n\n\n".join(
            f"#{res.title}\n\n{res.content}" for res in response.results
        )
        return results_as_text


SEARCH_FUNCTIONS: list[type[Function]] = [
    TavilySearch,
]
