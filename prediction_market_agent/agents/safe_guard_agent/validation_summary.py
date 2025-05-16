from prediction_market_agent_tooling.config import APIKeys
from pydantic_ai import Agent

from prediction_market_agent.agents.safe_guard_agent.guards.llm import (
    format_transaction,
)
from prediction_market_agent.agents.safe_guard_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_guard_agent.validation_result import (
    ValidationResult,
)
from prediction_market_agent.tools.openai_utils import OpenAIModel, get_openai_provider


def create_validation_summary(
    tx: DetailedTransactionResponse, results: list[ValidationResult]
) -> str:
    agent = Agent(
        OpenAIModel(
            "gpt-4o",
            provider=get_openai_provider(api_key=APIKeys().openai_api_key),
        ),
        output_type=str,
        system_prompt="You are an expert in on-chain malicious activity observation.",
    )
    points = "\n".join(
        f"- {result.name} ({result.description}): {result.reason}" for result in results
    )
    result = agent.run_sync(
        f"""Given the following details about the transaction, write a short summary. 
Focus only on the malicious parts. 
If everything is ok, just write a brief summary that everything is ok and what the transaction is doing.
Cut the fluff and be concise.

Transaction:
{format_transaction(tx)}

Executed verifications:
{points}"""
    )
    llm_summary = result.output

    return f"""{llm_summary}

The following checks were performed:
{points}
"""
