from prediction_market_agent_tooling.tools.google import search_google

search_google_schema = {
    "type": "function",
    "function": {
        "name": "search_google",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The google search query.",
                }
            },
            "required": ["query"],
        },
        "description": "Google search to return search results from a query.",
    },
}


class GoogleSearchTool:
    def __init__(self) -> None:
        self.fn = search_google
        self.schema = search_google_schema
