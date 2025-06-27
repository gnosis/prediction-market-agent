from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.tools.is_invalid import QUESTION_IS_INVALID_PROMPT
from prediction_market_agent_tooling.tools.utils import (
    LLM_SUPER_LOW_TEMPERATURE,
    utcnow,
)

from prediction_market_agent.utils import APIKeys

REPHRASE_PROMPT = """
Today is {current_date}.
The question below was marked "invalid" by another LLM that had the prompt below. Rephrase the question into a "valid" question.
Output only the rephrased question and nothing else.

[QUESTION]
"{question}"

[PROMPT FOR QUESTION VALIDITY]
{invalid_question_prompt}
"""


def rephrase(question: str, engine: str = "gpt-4.1-mini-2025-04-14") -> str:
    llm = ChatOpenAI(
        model_name=engine,
        temperature=LLM_SUPER_LOW_TEMPERATURE,
        openai_api_key=APIKeys().openai_api_key,
    )

    prompt = ChatPromptTemplate.from_template(template=REPHRASE_PROMPT)
    messages = prompt.format_messages(
        question=question,
        current_date=str(utcnow()),
        invalid_question_prompt=QUESTION_IS_INVALID_PROMPT,
    )
    completion = str(
        llm.invoke(
            messages,
            max_tokens=1024,
        ).content
    )

    return completion
