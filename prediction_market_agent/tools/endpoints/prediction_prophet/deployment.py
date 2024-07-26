import math

from fastapi import Request, Response
from langchain_community.callbacks import get_openai_callback
from modal import App, Secret, web_endpoint
from prediction_prophet.benchmark.agents import (
    _make_prediction as prophet_make_prediction,
)
from prediction_prophet.functions.research import research as prophet_research
from pydantic import BaseModel

from prediction_market_agent.tools.endpoints.utils import MODAL_IMAGE
from prediction_market_agent.utils import APIKeys

MODEL = "gpt-4o"
keys = APIKeys()


class PredictionProphetResponse(BaseModel):
    p_yes: float
    cost_usd: float


def usd_to_credits(cost_usd: float) -> int:
    # 1 USD = 100 credits
    return math.ceil(cost_usd * 100)


def predict(question: str) -> Response:
    try:
        with get_openai_callback() as openai_token_tracker:
            research = prophet_research(
                goal=question,
                use_summaries=False,
                model=MODEL,
                openai_api_key=keys.openai_api_key,
                tavily_api_key=keys.tavily_api_key,
            )
            prediction = prophet_make_prediction(
                market_question=question,
                additional_information=research,
                engine=MODEL,
                temperature=0,
                api_key=keys.openai_api_key,
            )

        if prediction.outcome_prediction is None:
            raise ValueError("Failed to make a prediction.")

        # TODO calculate dynamically
        # From https://tavily.com/#pricing:
        # 220 usd/month for 38000 search requests -> 0.0058 usd/request
        tavily_cost = 5 * 0.0058  # `prophet_research` makes 5 searches

        margin = 1.2
        cost_usd = (openai_token_tracker.total_cost + tavily_cost) * margin
        content = PredictionProphetResponse(
            p_yes=prediction.outcome_prediction.p_yes,
            cost_usd=cost_usd,
        )

        # Set the cost of the request via the header
        cost_credits = usd_to_credits(cost_usd)
        headers = {"NVMCreditsConsumed": str(cost_credits)}

        return Response(
            content=content.model_dump_json(),
            headers=headers,
            media_type="application/json",
        )
    except Exception as e:
        return Response(
            content={"Internal error": str(e)},
            status_code=500,
            headers={"NVMCreditsConsumed": str(0)},
            media_type="application/json",
        )


app = App(image=MODAL_IMAGE)


@app.function(secrets=[Secret.from_name("prophet")])
@web_endpoint(docs=True)
def predict_wrapper(request: Request) -> Response:
    query_params = dict(request.query_params)
    return predict(**query_params)
