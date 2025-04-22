import pytest
from pydantic import BaseModel
from prediction_market_agent.agents.ofvchallenger_agent.ofv_models import Factuality


class TestModel(BaseModel):
    factuality: Factuality


@pytest.mark.parametrize(
    "input_value, expected_value",
    [
        (None, None),
        (" NoNE ", None),
        ("nothing to check.", None),
        ("non-FACTUAL", None),
        ("non-factual claim", None),
        ("non-factual claiM", None),
    ],
)
def test_factuality_type(input_value: str | None, expected_value: bool | None) -> None:
    assert (
        TestModel.model_validate({"factuality": input_value}).factuality
        == expected_value
    )
