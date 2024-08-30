from factcheck import FactCheck
from factcheck.utils.multimodal import modal_normalization
from langchain_openai import ChatOpenAI
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.is_predictable import is_predictable_binary
from prediction_market_agent_tooling.tools.langfuse_ import (
    get_langfuse_langchain_config,
    observe,
)

from prediction_market_agent.agents.ofvchallenger_agent.ofv_models import (
    FackCheckAnswer,
    FactCheckResult,
    Factuality,
)
from prediction_market_agent.utils import APIKeys

DEFAULT_OPENAI_MODEL = "gpt-4-0125-preview"


@observe()
def factcheck(
    statement: str,
    api_keys: APIKeys,
    model: str = DEFAULT_OPENAI_MODEL,
) -> FactCheckResult:
    factcheck = FactCheck(
        default_model=model,
        api_config={
            "OPENAI_API_KEY": api_keys.openai_api_key.get_secret_value(),
            "SERPER_API_KEY": api_keys.serper_api_key.get_secret_value(),
        },
        retriever="serper",
        num_seed_retries=5,
    )
    content = modal_normalization("string", statement)
    res = factcheck.check_response(content)
    return FactCheckResult.model_validate(res)


@observe()
def rewrite_as_sentence(
    question: str,
    api_keys: APIKeys,
    model: str = DEFAULT_OPENAI_MODEL,
) -> str:
    """
    Rewrites the question into a sentence, example:
    `Will former Trump Organization CFO Allen Weisselberg be sentenced to jail by 15 April 2024?`
    ->
    `Former Trump Organization CFO Allen Weisselberg was sentenced to jail by 15 April 2024.`
    """
    llm = ChatOpenAI(
        model=model, temperature=0.0, api_key=api_keys.openai_api_key_secretstr_v1
    )

    prompt = f"""
Rewrite the question into a simple announcement sentence stating a fact or prediction like it is already known.  
Make future tense into past tense.
For future questions that ask if something will happen "by" some date, rewrite it to "before" that date or any time sooner.
For future questions that ask if something will happen "on" some date, rewrite it to "on" that date.
If the question is both "on" and "by" some date, rewrite it as "before or any time sooner than" that date.
If the question is about exact date, keep it exact. 
If the question is about a date range, keep it a range.
Always keep the same meaning.                          
Never negate the sentence into opposite meaning of the question.                  
Question: {question}
Sentence:                                         
"""
    completion = str(
        llm.invoke(
            prompt, max_tokens=512, config=get_langfuse_langchain_config()
        ).content
    )

    return completion


@observe()
def most_common_fact_result(
    results: list[FactCheckResult],
) -> tuple[Factuality, list[FactCheckResult]]:
    factualities = [fact.factuality for fact in results]
    most_common_factuality = max(set(factualities), key=factualities.count)
    results_with_most_common_factuality = [
        fact for fact in results if fact.factuality == most_common_factuality
    ]
    return most_common_factuality, results_with_most_common_factuality


@observe()
def ofv_answer_binary_question(
    market_question: str,
    api_keys: APIKeys,
    n_fact_runs: int = 3,
) -> FackCheckAnswer | None:
    """
    Run the prediction market resolver based on Open Fact Verifier.
    """
    assert (
        n_fact_runs > 0 and n_fact_runs % 2 != 0
    ), "n_fact_runs must be greater than 0 and an odd number"

    # Check if the question is reasonable to look for an answer.
    is_answerable = is_predictable_binary(market_question)
    if not is_answerable:
        logger.warning(
            f"Question `{market_question}` is not answerable, skipping fact checking."
        )
        return None

    # Rewrite the question (which was about a future) into a sentence (which is about the past).
    market_sentence = rewrite_as_sentence(market_question, api_keys)
    logger.info(f"Question `{market_question}` rewritten into `{market_sentence}`.")

    # Fact-check the sentence.
    factresults = [factcheck(market_sentence, api_keys) for _ in range(n_fact_runs)]
    (
        most_common_factuality,
        factresults_with_most_common_factuality,
    ) = most_common_fact_result(factresults)
    logger.info(
        f"Fact check result for `{market_sentence}` is `{most_common_factuality}` because {factresults_with_most_common_factuality[0].claims_details}."
    )

    return FackCheckAnswer(
        factuality=most_common_factuality,
        chosen_results=factresults_with_most_common_factuality,
        all_considered_results=factresults,
    )
