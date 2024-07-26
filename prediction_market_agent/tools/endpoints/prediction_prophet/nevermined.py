from typing import Dict, TypeAlias

from payments_py import Environment, Payments

from prediction_market_agent.tools.endpoints.nevermined_utils import (
    NeverminedSettings,
    create_service,
)

Headers: TypeAlias = Dict[str, str]

NAME = "Prediction Prophet"
DESCRIPTION = (
    "Ask a yes/no question about a future event, and get a prediction of the "
    "probability of a 'yes' outcome. Uses Tavily web search and LLMs to perform "
    "research, and turn the research into a prediction."
)

# Service costs between 0 and 20 credits (i.e. between $0.01 and $0.20)
MIN_SERVICE_CHARGE = 0
MAX_SERVICE_CHARGE = 20
FLAT_SERVICE_CHARGE = 10

SERVICE_ENDPOINT = "https://evangriffiths--prediction-prophet-predict-wrapper.modal.run"
SERVICE_OPENAPI_URL = f"{SERVICE_ENDPOINT}/openapi.json"

if __name__ == "__main__":
    nevermined_settings = NeverminedSettings()
    payments = Payments(
        nvm_api_key=nevermined_settings.CREATOR_API_KEY,
        environment=Environment.appTesting,
    )

    # Create service
    service_did = create_service(
        payments=payments,
        name=NAME,
        description=DESCRIPTION,
        endpoint_url=SERVICE_ENDPOINT,
        open_api_url=SERVICE_OPENAPI_URL,
        min_credits_to_charge=MIN_SERVICE_CHARGE,
        max_credits_to_charge=MAX_SERVICE_CHARGE,
    )

    print("Service created successfully. ID: ", service_did)
