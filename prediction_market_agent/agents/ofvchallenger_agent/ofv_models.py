from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator

Factuality = Annotated[
    bool | None,
    BeforeValidator(
        lambda v: (
            None
            if str(v).lower().strip() in ("Nothing to check.", "non-factual", "none")
            else v
        )
    ),
]


class FactCheckClaimDetails(BaseModel):
    claim: str
    factuality: Factuality
    correction: str | dict[str, Any] | None = None
    reference_url: str


class FactCheckResult(BaseModel):
    factuality: Factuality
    claims_details: list[FactCheckClaimDetails] | None = None


class FactCheckAnswer(BaseModel):
    factuality: Factuality
    chosen_results: list[FactCheckResult]
    all_considered_results: list[FactCheckResult]
