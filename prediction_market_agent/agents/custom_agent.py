import json
from typing import Optional

import requests
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.tools.utils import (
    check_not_none,
    should_not_happen,
)

from prediction_market_agent import utils
from prediction_market_agent.agents.abstract import AbstractAgent
from prediction_market_agent.ai_models.abstract_ai_models import (
    AbstractAiChatModel,
    Message,
)
from prediction_market_agent.ai_models.llama_ai_models import ChatReplicateLLamaModel
from prediction_market_agent.ai_models.openai_ai_models import ChatOpenAIModel
from prediction_market_agent.tools.tool_exception_handler import tool_exception_handler
from prediction_market_agent.tools.web_scrape.structured_summary import (
    web_scrape_structured_and_summarized,
)
from prediction_market_agent.tools.web_search.google import google_search


class CustomAgent(AbstractAgent):
    SYSTEM_PROMPT = f"""You are a research assistant meant to answer questions placed on a predictive market.

To do that, you can't rely on your training data, but you have access to the following tools in order to answer question:

1. GoogleSearchTool: Given a query, returns a list of URLs. Accepts arguments as a dictionary, with a single key "query".
2. WebScrapingTool: Given an objective and url, returns the summarized content of the page. In the objective, ask closed-ended questions. Accepts arguments as a dictionary, with keys "objective" and "url".

If you want to use the tool, simply return completion in form of a dictionary with the keys "tool_name" and "tool_params".

If you used a tool in the past, its output will be available in a message in form of a dictionary with the keys "tool_name" and "tool_output".

If you need to think about the question, return a completion in form of a dictionary with a single key "thinking" and a string value.

If you aren't sure, feel free to continue research by using the tools and following URLs until you have all the information, unless you are told otherwise.

If you want to answer, return a completion in form of a dictionary with a single key "answer" and a string value.
"""

    def __init__(self, model: AbstractAiChatModel, max_cycles: int, verbose: bool):
        """
        Initialize the CustomAgent.

        Args:
            verbose: Whether to display intermediate thinking steps.
            max_cycles: The maximum number of thinking steps for the agent.
        """

        self.verbose = verbose
        self.max_cycles = max_cycles
        self.keys = utils.APIKeys()
        self.google_search = google_search
        self.web_scrap_and_summarize = tool_exception_handler(
            map_exception_to_output={
                requests.exceptions.HTTPError: "Couldn't reach the URL. Don't scrape the same domain again."
            }
        )(web_scrape_structured_and_summarized)
        self.model = model

    def verbose_log(self, message: str) -> None:
        if self.verbose:
            logger.info(f"{message}\n")

    def answer_binary_market(self, market: AgentMarket) -> bool:
        cycle_count = 0
        answer: Optional[str] = None
        # Get the main objective prompt.
        objective = utils.get_market_prompt(market.question)
        # Keep track of history of messages.
        messages: list[Message] = [Message(role="user", content=objective)]

        while True:
            cycle_count += 1
            if cycle_count == self.max_cycles - 1:
                # If we reached the maximum number of cycles, ask to provide an answer based on available information.
                self.verbose_log(
                    f"Agent: Reached {cycle_count} cycles, asking to provide answer now."
                )
                messages.append(
                    Message(
                        role="user",
                        content="You have reached maximal message count, please provide answer now as best as you can.",
                    )
                )
            elif cycle_count >= self.max_cycles:
                # If model didn't follow the final answer prompt, raise an error.
                raise ValueError(
                    f"Reached {cycle_count} cycles, but no answer was given."
                )

            # Because of the system prompt, completions are expected to be JSON-parseable.
            completion_str = check_not_none(
                self.model.complete(messages), "Couldn't complete the prompt."
            )
            self.verbose_log(f"Completion: {completion_str}")
            try:
                completion_dict = json.loads(completion_str)
            except json.decoder.JSONDecodeError:
                raise ValueError(f"Can not parse JSON: {completion_str}")

            # Because of the system prompt, either:
            # - thinking: Chain of thoughts to be processed by the agent.
            # - tool_name: Name of the tool to be used.
            # - answer: Final answer to the question.
            if thinking := completion_dict.get("thinking"):
                self.verbose_log(f"Agent: Thinking about {thinking=}.")
                messages.append(
                    Message(
                        role="assistant", content=json.dumps({"thinking": thinking})
                    )
                )

            elif tool_name := completion_dict.get("tool_name"):
                tool_params = completion_dict.get("tool_params")
                self.verbose_log(f"Agent: Using {tool_name=} with {tool_params=}.")
                tool_output = (  # type: ignore # Untyped for the sake of simplicity when the LLM is used.
                    self.google_search
                    if tool_name == "GoogleSearchTool"
                    else (
                        self.web_scrap_and_summarize
                        if tool_name == "WebScrapingTool"
                        else should_not_happen("Unknown tool requested from the LLM.")
                    )
                )(
                    **tool_params
                )
                self.verbose_log(f"Tool: {tool_name=} returns {tool_output=}.")
                messages.append(
                    Message(
                        role="assistant",
                        content=json.dumps(
                            {"tool_name": tool_name, "tool_output": tool_output}
                        ),
                    )
                )

            elif answer := completion_dict.get("answer"):
                self.verbose_log(f"Agent: Answering {answer=}.")
                messages.append(
                    Message(role="assistant", content=json.dumps({"answer": answer}))
                )
                break

            else:
                raise ValueError(f"Can not process the completion: {completion_str}")

        assert answer is not None, "No answer was given."
        return utils.parse_result_to_boolean(answer)


class CustomAgentOpenAi(CustomAgent):
    def __init__(self, verbose: bool = True, max_cycles: int = 10):
        super().__init__(
            model=ChatOpenAIModel(
                model="gpt-4",  # I tried gpt-3.5-turbo, but it fails at providing correct inputs to the tools.
                system_prompt=CustomAgent.SYSTEM_PROMPT,
            ),
            verbose=verbose,
            max_cycles=max_cycles,
        )


class CustomAgentLlama(CustomAgent):
    def __init__(self, verbose: bool = True, max_cycles: int = 10):
        super().__init__(
            model=ChatReplicateLLamaModel(
                model="meta/llama-2-70b-chat",
                system_prompt=CustomAgent.SYSTEM_PROMPT,
            ),
            verbose=verbose,
            max_cycles=max_cycles,
        )
