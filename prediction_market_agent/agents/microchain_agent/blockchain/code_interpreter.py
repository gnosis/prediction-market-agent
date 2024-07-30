from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from prediction_market_agent.utils import APIKeys


class FunctionSummary(BaseModel):
    function_name: str = Field(..., description="Name of the function or property")
    summary: str = Field(..., description="The summary assigned to this function.")


class Summaries(BaseModel):
    """Identifying information about functions/properties of the source code."""

    summaries: list[FunctionSummary]


class CodeInterpreter:
    prompt_and_model: RunnableSerializable[dict[Any, Any], Any]

    def __init__(
        self, source_code: str, summarization_model: str = "gpt-4-turbo"
    ) -> None:
        self.summarization_model = summarization_model
        self.source_code = source_code
        self.keys = APIKeys()
        self.build_chain()

    def build_chain(self) -> None:
        parser = PydanticOutputParser(pydantic_object=Summaries)

        prompt = PromptTemplate(
            template="""Generate summaries of the functions given by FUNCTION_NAMES. The function definitions can be found
            in the SOURCE_CODE, which contains lines of codes written in Solidity.

            [FUNCTION_NAMES]
            {function_names}

            [SOURCE_CODE]
            {source_code}

            {format_instructions}
            """,
            input_variables=["function_names", "source_code"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        llm = ChatOpenAI(
            temperature=0,
            model=self.summarization_model,
            api_key=APIKeys().openai_api_key_secretstr_v1,
        )

        self.prompt_and_model = prompt | llm | parser

    def generate_summary(self, function_names: list[str]) -> Summaries:
        summaries: Summaries = self.prompt_and_model.invoke(
            {
                "function_names": function_names,
                "source_code": self.source_code,
            }
        )
        return summaries
