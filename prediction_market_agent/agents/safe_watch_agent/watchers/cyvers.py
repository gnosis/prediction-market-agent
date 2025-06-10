from datetime import timedelta
from typing import Any

import requests
from prediction_market_agent_tooling.gtypes import ChainID, ChecksumAddress, HexStr
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.caches.db_cache import db_cache
from prediction_market_agent_tooling.tools.langfuse_ import observe
from safe_eth.safe.safe import SafeTx

from prediction_market_agent.agents.safe_watch_agent.safe_api_models.detailed_transaction_info import (
    DetailedTransactionResponse,
)
from prediction_market_agent.agents.safe_watch_agent.validation_result import (
    ValidationResult,
)
from prediction_market_agent.agents.safe_watch_agent.watchers.abstract_watch import (
    AbstractWatch,
)
from prediction_market_agent.agents.safe_watch_agent.watchers.goplus_ import (
    _build_validation_result,
)
from prediction_market_agent.tools.streamlit_utils import dict_to_point_list
from prediction_market_agent.utils import APIKeys


class CyversAddressReputation(AbstractWatch):
    name = "Cyvers address reputation"
    description = "This watch checks the reputation of address via Cyvers API."

    @observe(name="validate_cyvers_address_reputation")
    def validate(
        self,
        new_transaction: DetailedTransactionResponse,
        new_transaction_safetx: SafeTx,
        all_addresses_from_tx: list[ChecksumAddress],
        history: list[DetailedTransactionResponse],
        chain_id: ChainID,
    ) -> ValidationResult | None:
        malicious_reasons: list[str] = []

        for addr in all_addresses_from_tx:
            result_for_addr = cyvers_address_reputation(addr)
            if result_for_addr is None:
                continue

            logger.info(
                f"{self.name} response for address {addr}:\n\n{dict_to_point_list(result_for_addr)}",
                streamlit=True,
            )

            if (risk_score := result_for_addr["data"]["risk_score"].lower()) != "low":
                malicious_reasons.append(f"Address {addr} has risk score {risk_score}.")

        return _build_validation_result(self.name, self.description, malicious_reasons)


class CyversSimulateSafeTX(AbstractWatch):
    name = "Cyvers Safe TX simulation"
    description = "This simulates Safe transactions using Cyvers API."

    @observe(name="validate_cyvers_simulate_safe_tx")
    def validate(
        self,
        new_transaction: DetailedTransactionResponse,
        new_transaction_safetx: SafeTx,
        all_addresses_from_tx: list[ChecksumAddress],
        history: list[DetailedTransactionResponse],
        chain_id: ChainID,
    ) -> ValidationResult | None:
        malicious_reasons: list[str] = []
        simulation_response = cyvers_simulate_safe_tx(
            HexStr(new_transaction_safetx.safe_tx_hash.hex()), chain_id
        )

        risk_level = simulation_response["data"]["risk_level"]
        risk_description = simulation_response["data"]["risk_description"]

        if risk_level.lower() != "low":
            malicious_reasons.append(
                f"Safe TX has a risk level {risk_level}, because of {risk_description}."
            )

        return _build_validation_result(self.name, self.description, malicious_reasons)


@observe()
@db_cache(max_age=timedelta(days=3))
def cyvers_address_reputation(address: str) -> dict[str, Any]:
    url = f"https://vigilens-api.cyvers.ai/v1/risk-score/{address}"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {APIKeys().cyvers_api_key.get_secret_value()}",
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response_json: dict[str, Any] | None = response.json()
    if response_json is None:
        raise ValueError(f"Invalid output `None` from {url}.")
    return response_json


@observe()
@db_cache(max_age=timedelta(days=3))
def cyvers_simulate_safe_tx(
    safe_tx_hash: HexStr,
    chain_id: ChainID,
) -> dict[str, Any]:
    url = "https://vigilens-api.cyvers.ai/v1/tx-simulation"
    payload = {
        "safe_tx_hash": safe_tx_hash,
        "chain_id": chain_id,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {APIKeys().cyvers_api_key.get_secret_value()}",
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    response_json: dict[str, Any] | None = response.json()
    if response_json is None:
        raise ValueError(f"Invalid output `None` from {url}.")
    return response_json
