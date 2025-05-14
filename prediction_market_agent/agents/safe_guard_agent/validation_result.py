from pydantic import BaseModel


class ValidationResult(BaseModel):
    name: str
    description: str
    reason: str
    ok: bool


class ValidationConclusion(BaseModel):
    all_ok: bool
    summary: str
    results: list[ValidationResult]
