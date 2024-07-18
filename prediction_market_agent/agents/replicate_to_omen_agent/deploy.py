from datetime import timedelta

from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import (
    redeem_from_all_user_positions,
)
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from web3 import Web3

from prediction_market_agent.agents.replicate_to_omen_agent.omen_replicate import (
    omen_replicate_from_tx,
    omen_unfund_replicated_known_markets_tx,
)
from prediction_market_agent.agents.replicate_to_omen_agent.omen_resolve_replicated import (
    omen_finalize_and_resolve_and_claim_back_all_markets_based_on_others_tx,
)
from prediction_market_agent.utils import APIKeys

REPLICATOR_ADDRESS = Web3.to_checksum_address(
    "0x993DFcE14768e4dE4c366654bE57C21D9ba54748"
)


class ReplicateConfig(BaseModel):
    n: int  # How many markets of this configuration to replicate.
    close_time_up_to_n_days: (
        int  # Consider only markets closing in less than N days for this configuration.
    )
    every_n_days: int  # This configuration should execute every N days.
    sources: list[MarketType] = [MarketType.MANIFOLD, MarketType.POLYMARKET]


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

        keys = APIKeys()
        settings = ReplicateSettings()
        now = utcnow()

        logger.info(
            f"Finalising, resolving and claiming back xDai from existing markets replicated by {keys.bet_from_address}."
        )
        omen_finalize_and_resolve_and_claim_back_all_markets_based_on_others_tx(keys)

        logger.info(
            f"Unfunding soon to be known markets replicated by {keys.bet_from_address}."
        )
        omen_unfund_replicated_known_markets_tx(keys, saturation_above_threshold=0.9)

        logger.info("Redeeming funds from previously unfunded markets.")
        redeem_from_all_user_positions(keys)

        for replicate_config in settings.REPLICATE:
            if now.timetuple().tm_yday % replicate_config.every_n_days:
                logger.info(f"Skipping {replicate_config}, because it's not his day.")
                continue

            to_replicate = replicate_config.n

            for replicate_from_market_type in replicate_config.sources:
                close_time_before = now + timedelta(
                    days=replicate_config.close_time_up_to_n_days
                )
                initial_funds_per_market = xdai_type(settings.INITIAL_FUNDS)

                logger.info(
                    f"Replicating {to_replicate} from {replicate_from_market_type} markets closing in {replicate_config.close_time_up_to_n_days} days."
                )
                replicated = omen_replicate_from_tx(
                    market_type=replicate_from_market_type,
                    n_to_replicate=to_replicate,
                    initial_funds=initial_funds_per_market,
                    api_keys=keys,
                    close_time_before=close_time_before,
                    auto_deposit=True,
                )
                to_replicate -= len(replicated)

                if to_replicate <= 0:
                    break

            logger.info(f"Replication from {replicate_from_market_type} done.")

        logger.info("All done.")
