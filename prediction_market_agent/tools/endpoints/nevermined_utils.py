from typing import Dict, Tuple, TypeAlias

from payments_py import Payments
from prediction_market_agent_tooling.loggers import logger
from pydantic_settings import BaseSettings, SettingsConfigDict

Headers: TypeAlias = Dict[str, str]

# Standardisation of subscriptions:
# A subscription costs $1, and gives 100 credits
# A service creator can decide how many credits to charge for their service
SUBSCRIPTION_COST = 1000000  # 1 USDC
SUBSCRIPTION_CREDITS = 100


class NeverminedSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    CREATOR_API_KEY: str
    CONSUMER_API_KEY: str
    CONSUMER_ADDRESS: str


def get_subscription_balance(
    payments: Payments,
    account_address: str,
    subscription_did: str,
) -> int:
    response = payments.get_subscription_balance(
        subscription_did=subscription_did,
        account_address=account_address,
    )
    response.raise_for_status()
    return int(response.json()["balance"])


def service_did_from_subscription(payments: Payments, subscription_did: str) -> str:
    response = payments.get_subscription_associated_services(subscription_did)
    response.raise_for_status()
    response_json = response.json()
    if len(response_json) != 1:
        raise ValueError(f"Expected 1 service, got {len(response_json)}")
    return response.json()[0]


def get_endpoint_and_headers(
    payments: Payments,
    service_did: str,
) -> Tuple[str, Headers]:
    service_token_response = payments.get_service_token(service_did)
    service_token_response.raise_for_status()
    response_json = service_token_response.json()
    jwt_token = response_json["token"]["accessToken"]
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Content-Type": "application/json",
    }
    endpoint = response_json["token"]["neverminedProxyUri"]
    return endpoint, headers


def create_subscription(
    payments: Payments,
    name: str,
    tags: list[str] = [],
) -> str:
    subscription_response = payments.create_subscription(
        name=name,
        description=f"A subscription for the {name} service",
        price=SUBSCRIPTION_COST,
        token_address="0x75faf114eafb1BDbe2F0316DF893fd58CE46AA4d",  # USDC
        amount_of_credits=SUBSCRIPTION_CREDITS,
        duration=100000,  # TODO how to make 'forever'?
        tags=tags,
    )
    subscription_response.raise_for_status()
    return subscription_response.json()["did"]


def create_service(
    payments: Payments,
    name: str,
    description: str,
    endpoint_url: str,
    open_api_url: str,
    min_credits_to_charge: int,
    max_credits_to_charge: int,
) -> str:
    logger.info("Creating subscription...")
    subscription_did = create_subscription(payments=payments, name=name)
    logger.info("Subscription created successfully. ID: ", subscription_did)

    logger.info("Creating service...")
    service_response = payments.create_service(
        subscription_did=subscription_did,
        name=name,
        description=description,
        service_charge_type="dynamic",
        auth_type="none",
        endpoints=[{"get": endpoint_url}],
        open_api_url=open_api_url,
        min_credits_to_charge=min_credits_to_charge,
        max_credits_to_charge=max_credits_to_charge,
        amount_of_credits=0,  # Placeholder, unused TODO
    )
    service_response.raise_for_status()
    return service_response.json()["did"]


def topup_if_required(
    payments: Payments,
    account_address: str,
    subscription_did: str,
    min_credits: int,
) -> None:
    # TODO check balance of `CONSUMER_ADDRESS`, and top up from agent's main
    # wallet if necessary.

    init_balance = get_subscription_balance(
        payments=payments,
        account_address=account_address,
        subscription_did=subscription_did,
    )

    balance = init_balance
    while balance < min_credits:
        logger.info("Topping up...")
        order_response = payments.order_subscription(subscription_did=subscription_did)
        order_response.raise_for_status()
        new_balance = get_subscription_balance(
            payments=payments,
            account_address=account_address,
            subscription_did=subscription_did,
        )
        assert new_balance > init_balance
        balance = new_balance
