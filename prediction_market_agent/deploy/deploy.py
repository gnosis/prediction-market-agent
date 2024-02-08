from cron_validator import CronValidator
from enum import Enum
import os
from pydantic import BaseModel
import requests
import shutil
import subprocess
import tempfile
import time

from prediction_market_agent.data_models.market_data_models import AgentMarket
from prediction_market_agent.deploy.utils import (
    export_requirements_from_toml,
    gcloud_delete_function_cmd,
    gcloud_deploy_cmd,
    gcloud_schedule_cmd,
    get_gcloud_function_uri,
    get_gcloud_id_token,
)
from prediction_market_agent.markets.all_markets import (
    MarketType,
    get_binary_markets,
    place_bet,
)
from prediction_market_agent.tools.betting_strategies import get_tiny_bet
from prediction_market_agent.utils import APIKeys


class DeploymentType(str, Enum):
    GOOGLE_CLOUD = "google_cloud"
    LOCAL = "local"


class DeployableAgent(BaseModel):
    def pick_markets(self, markets: list[AgentMarket]) -> list[AgentMarket]:
        return markets[:1]

    def answer_binary_market(self, market: AgentMarket) -> bool:
        raise NotImplementedError("This method should be implemented by the subclass")

    def deploy(
        self,
        sleep_time: int,
        market_type: MarketType,
        deployment_type: DeploymentType,
        api_keys: APIKeys,
    ) -> None:
        if deployment_type == DeploymentType.GOOGLE_CLOUD:
            # Deploy to Google Cloud Functions, and use Google Cloud Scheduler to run the function
            raise NotImplementedError("TODO not currently possible programatically")
        elif deployment_type == DeploymentType.LOCAL:
            while True:
                self.run(market_type, api_keys)
                time.sleep(sleep_time)

    def run(self, market_type: MarketType, api_keys: APIKeys) -> None:
        available_markets = [
            x.to_agent_market() for x in get_binary_markets(market_type)
        ]
        markets = self.pick_markets(available_markets)
        for market in markets:
            result = self.answer_binary_market(market)
            print(f"Placing bet on {market} with result {result}")
            place_bet(
                market=market.original_market,
                amount=get_tiny_bet(market_type),
                outcome=result,
                keys=api_keys,
                omen_auto_deposit=True,
            )

    @classmethod
    def get_gcloud_fname(cls, market_type: MarketType) -> str:
        return f"{cls.__class__.__name__.lower()}-{market_type}-{int(time.time())}"


def deploy_to_gcp(
    function_file: str,
    requirements_file: str,
    extra_deps: list[str],
    api_keys: dict[str, str],
    market_type: MarketType,
    memory: int,  # in MB
) -> str:
    if not os.path.exists(requirements_file):
        raise ValueError(f"File {requirements_file} does not exist")

    if not os.path.exists(function_file):
        raise ValueError(f"File {function_file} does not exist")

    gcp_fname = DeployableAgent().get_gcloud_fname(market_type=market_type)

    # Make a tempdir to store the requirements file and the function
    with tempfile.TemporaryDirectory() as tempdir:
        # Copy function_file to tempdir/main.py
        shutil.copy(function_file, f"{tempdir}/main.py")

        # If the file is a .toml file, convert it to a requirements.txt file
        if requirements_file.endswith(".toml"):
            export_requirements_from_toml(output_dir=tempdir, extra_deps=extra_deps)
        else:
            shutil.copy(requirements_file, f"{tempdir}/requirements.txt")

        # Deploy the function
        cmd = gcloud_deploy_cmd(
            gcp_function_name=gcp_fname,
            source=tempdir,
            entry_point="main",  # TODO check this function exists in main.py
            api_keys=api_keys,
            memory=memory,
        )
        subprocess.run(cmd, shell=True)
        # TODO test the depolyment without placing a bet

    return gcp_fname


def schedule_deployed_gcp_function(function_name: str, cron_schedule: str) -> None:
    # Validate the cron schedule
    if not CronValidator().parse(cron_schedule):
        raise ValueError(f"Invalid cron schedule {cron_schedule}")

    cmd = gcloud_schedule_cmd(function_name=function_name, cron_schedule=cron_schedule)
    subprocess.run(cmd, shell=True)


def run_deployed_gcp_function(function_name: str) -> requests.Response:
    uri = get_gcloud_function_uri(function_name)
    header = {"Authorization": f"Bearer {get_gcloud_id_token()}"}
    return requests.post(uri, headers=header)


def remove_deployed_gcp_function(function_name: str) -> None:
    cmd = gcloud_delete_function_cmd(function_name)
    subprocess.run(cmd, shell=True)
