from typing import Annotated

from pydantic import BaseModel, BeforeValidator

Factuality = Annotated[
    bool | None,
    BeforeValidator(lambda v: None if v in ("Nothing to check.", "non-factual") else v),
]


class FactCheckClaimDetails(BaseModel):
    claim: str
    factuality: Factuality
    correction: str | None
    reference_url: str


class FactCheckResult(BaseModel):
    factuality: Factuality
    claims_details: list[FactCheckClaimDetails] | None


class FackCheckAnswer(BaseModel):
    factuality: Factuality
    chosen_results: list[FactCheckResult] | None
    all_considered_results: list[FactCheckResult] | None
