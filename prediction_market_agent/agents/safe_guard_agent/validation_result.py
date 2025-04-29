from pydantic import BaseModel


class ValidationResult(BaseModel):
    reason: str
    ok: bool


class ValidationResultWithName(ValidationResult):
    name: str

    @staticmethod
    def from_result(result: ValidationResult, name: str) -> "ValidationResultWithName":
        return ValidationResultWithName(
            reason=result.reason,
            ok=result.ok,
            name=name,
        )


class ValidationConclusion(BaseModel):
    all_ok: bool
    results: list[ValidationResultWithName]
