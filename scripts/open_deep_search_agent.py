import asyncio
import os

import nest_asyncio
from opendeepsearch import OpenDeepSearchAgent, OpenDeepSearchTool
from smolagents import LiteLLMModel, CodeAgent

from prediction_market_agent.utils import APIKeys


def dummy(
    search_tool: OpenDeepSearchAgent,
    query: str,
    max_sources: int = 2,
    pro_mode: bool = False,
):
    try:
        # Try getting the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in a running event loop (e.g., Jupyter), use nest_asyncio
            nest_asyncio.apply()
    except RuntimeError:
        # If there's no event loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(
        search_tool.search_and_build_context(query, max_sources, pro_mode)
    )


def main():
    keys = APIKeys()
    # Set environment variables for API keys
    os.environ["SERPER_API_KEY"] = keys.serper_api_key.get_secret_value()
    os.environ["OPENROUTER_API_KEY"] = keys.openrouter_api_key.get_secret_value()
    os.environ[
        "JINA_API_KEY"
    ] = "jina_fd4f27c2763e492db6f2227b66e176acndGCWq4L0o6C0OST_sCILNyP7OIb"

    search_agent = OpenDeepSearchTool(
        model_name="openrouter/google/gemini-2.0-flash-001", reranker="jina"
    )  # Set pro_mode for deep search
    # # Set reranker to "jina", or "infinity" for self-hosted reranking
    # query = "Give me a probability, between 0 and 1, that Yoon is not the president of South Korea anytime before May"
    if not search_agent.is_initialized:
        search_agent.setup()
    # # result = search_agent.forward(query)
    # context = dummy(search_agent, query, max_sources=5, pro_mode=True)(
    #     query=query, max_sources=5, pro_mode=True
    # )
    # print(context)
    model = LiteLLMModel("openrouter/google/gemini-2.0-flash-001", temperature=0.2)
    code_agent = CodeAgent(tools=[search_agent], model=model)
    query = "How long would a cheetah at full speed take to run the length of Pont Alexandre III?"
    result = code_agent.run(query)
    print(result)


if __name__ == "__main__":
    print("start")
    main()
    print("finish")
