from typing import Annotated

from pydantic import BaseModel, BeforeValidator

Factuality = Annotated[
    bool | None,
    BeforeValidator(
        lambda v: (
            None if v in ("Nothing to check.", "non-factual", "Non-factual") else v
        )
    ),
]


class FactCheckClaimDetails(BaseModel):
    claim: str
    factuality: Factuality
    correction: str | None = None
    reference_url: str


class FactCheckResult(BaseModel):
    factuality: Factuality
    claims_details: list[FactCheckClaimDetails] | None = None


class FactCheckAnswer(BaseModel):
    factuality: Factuality
    chosen_results: list[FactCheckResult]
    all_considered_results: list[FactCheckResult]
