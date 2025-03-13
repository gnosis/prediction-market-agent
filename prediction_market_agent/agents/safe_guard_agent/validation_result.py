from pydantic import BaseModel


class ValidationResult(BaseModel):
    reason: str
    ok: bool
