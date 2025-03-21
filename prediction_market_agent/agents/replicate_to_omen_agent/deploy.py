from datetime import timedelta

from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.gtypes import USD, xDai
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from web3 import Web3

from prediction_market_agent.agents.replicate_to_omen_agent.omen_replicate import (
    omen_replicate_from_tx,
    omen_unfund_replicated_known_markets_tx,
)
from prediction_market_agent.agents.replicate_to_omen_agent.omen_resolve_replicated import (
    omen_finalize_and_resolve_and_claim_back_all_replicated_markets_tx,
)
from prediction_market_agent.utils import APIKeys

REPLICATOR_EOA_ADDRESS = Web3.to_checksum_address(
    "0xEdAf63b6dDc1c2B611B7539ec23B50213D4dCA38"
)
REPLICATOR_SAFE_ADDRESS = Web3.to_checksum_address(
    "0x55D8Dfc1e6F994079A6A7fdb9D7a2712dc1b87B2"
)
REPLICATOR_TAG = "replicator"
REPLICATOR_BOND = xDai(10)


class ReplicateConfig(BaseModel):
    n: int  # How many markets of this configuration to replicate.
    close_time_up_to_n_days: (
        int  # Consider only markets closing in less than N days for this configuration.
    )
    every_n_days: int  # This configuration should execute every N days.
    source: MarketType


class ReplicateSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    REPLICATE: list[ReplicateConfig]
    INITIAL_FUNDS: str


class DeployableReplicateToOmenAgent(DeployableAgent):
    def run(self, market_type: MarketType = MarketType.MANIFOLD) -> None:
        if market_type != MarketType.OMEN:
            raise RuntimeError("Can replicate only into Omen.")

        settings = ReplicateSettings()
        self.replicate(settings)

    @observe()
    def replicate(self, settings: ReplicateSettings) -> None:
        self.langfuse_update_current_trace(tags=[REPLICATOR_TAG])

        keys = APIKeys()
        now = utcnow()

        # Unfund markets as the first thing, to get back resources that we can use later in this script.
        logger.info(
            f"Unfunding soon to be known markets replicated by {keys.bet_from_address}."
        )
        omen_unfund_replicated_known_markets_tx(keys, saturation_above_threshold=0.9)

        logger.info(
            f"Finalising, resolving and claiming back xDai from existing markets replicated by {keys.bet_from_address}."
        )
        omen_finalize_and_resolve_and_claim_back_all_replicated_markets_tx(
            keys, realitio_bond=REPLICATOR_BOND
        )

        for replicate_config in settings.REPLICATE:
            if now.timetuple().tm_yday % replicate_config.every_n_days:
                logger.info(f"Skipping {replicate_config}, because it's not his day.")
                continue

            close_time_before = now + timedelta(
                days=replicate_config.close_time_up_to_n_days
            )
            initial_funds_per_market = USD(settings.INITIAL_FUNDS)

            logger.info(
                f"Replicating {replicate_config.n} from {replicate_config.source} markets closing in {replicate_config.close_time_up_to_n_days} days."
            )
            omen_replicate_from_tx(
                market_type=replicate_config.source,
                n_to_replicate=replicate_config.n,
                initial_funds=initial_funds_per_market,
                api_keys=keys,
                close_time_before=close_time_before,
                auto_deposit=True,
            )

            logger.info(f"Replication from {replicate_config.source} done.")

        logger.info("All done.")
