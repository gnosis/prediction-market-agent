import pytest
import requests

from prediction_market_agent.tools.endpoints.prediction_prophet.deployment import (
    PredictionProphetResponse,
    predict,
)
from tests.endpoints.utils import UvicornServer, to_fastapi_app
from tests.utils import RUN_PAID_TESTS


@pytest.mark.skipif(not RUN_PAID_TESTS, reason="This test costs money to run.")
def test_predict() -> None:
    app = to_fastapi_app(predict)
    with UvicornServer(app) as server:
        response = requests.get(
            server.url, params={"question": "Will it rain in Berlin tomorrow?"}
        )
        response.raise_for_status()
        r = PredictionProphetResponse.model_validate_json(response.content)
        assert r.cost_usd > 0
        assert r.p_yes >= 0 and r.p_yes <= 1
