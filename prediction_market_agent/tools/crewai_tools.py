import os
from typing import Any, Type

from crewai_tools.tools.base_tool import BaseTool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from pydantic.v1 import BaseModel, Field
from pydantic.v1.types import SecretStr

from prediction_market_agent.utils import APIKeys


class TavilyDevToolSchema(BaseModel):
    """Input for TXTSearchTool."""

    search_query: str = Field(
        ..., description="Mandatory search query you want to use to search the internet"
    )


class TavilyDevTool(BaseTool):
    name: str = "Search the internet"
    # From Langchain's Tavily integration
    description: str = """"A search engine optimized for comprehensive, accurate, \
and trusted results. Useful for when you need to answer questions \
about current events or about recent information. \
Input should be a search query. \
If the user is asking about something that you don't know about, \
you should probably use this tool to see if that can provide any information."""
    args_schema: Type[BaseModel] = TavilyDevToolSchema

    def _run(
        self,
        search_query: str,
        **kwargs: Any,
    ) -> Any:
        keys = APIKeys()
        return TavilySearchAPIWrapper(
            tavily_api_key=SecretStr(os.environ["TAVILY_API_KEY"])
        ).results(query=search_query)
