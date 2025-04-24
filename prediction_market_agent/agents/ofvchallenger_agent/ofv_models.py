from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator

Factuality = Annotated[
    bool | None,
    BeforeValidator(
        lambda v: (
            None
            if str(v).lower().strip() in ["none"]
            or "non-factual" in str(v).lower()
            or "nothing to check" in str(v).lower()
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
