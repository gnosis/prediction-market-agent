from typing import Literal

import httpx
from prediction_market_agent_tooling.gtypes import ChainID
from pydantic import BaseModel
from web3.types import TxParams

from prediction_market_agent.agents.safe_guard_agent.tenderly.tenderly_config import (
    TenderlyKeys,
)


class SimulationInput(BaseModel, TxParams):
    network_id: ChainID
    simulation_type: Literal["quick", "full", "abi"]


class TenderlySimulator:
    def __init__(self, keys: TenderlyKeys):
        self.keys = keys

    def simulate_tx(self, data: SimulationInput):
        client = httpx.Client()
        url = f"https://api.tenderly.co/api/v1/account/{self.keys.TENDERLY_ACCOUNT_SLUG}/project/{self.keys.TENDERLY_PROJECT_SLUG}/simulate"

        headers = {"X-Access-Key": self.keys.TENDERLY_ACCESS_KEY.get_secret_value()}
        response = client.post(url, json=data.model_dump(), headers=headers)
        response.raise_for_status()
        return response
