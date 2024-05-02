from datetime import timedelta

from prediction_market_agent_tooling.config import APIKeys, PrivateCredentials
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.gtypes import xdai_type
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import (
    redeem_from_all_user_positions,
)
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic_settings import BaseSettings, SettingsConfigDict

from prediction_market_agent.agents.replicate_to_omen_agent.omen_replicate import (
    omen_replicate_from_tx,
    omen_unfund_replicated_known_markets_tx,
)
from prediction_market_agent.agents.replicate_to_omen_agent.omen_resolve_replicated import (
    omen_finalize_and_resolve_and_claim_back_all_markets_based_on_others_tx,
)


class ReplicateSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    N_TO_REPLICATE: int
    INITIAL_FUNDS: str
    CLOSE_TIME_UP_TO_N_DAYS: list[int]


class DeployableReplicateToOmenAgent(DeployableAgent):
    def run(
        self, market_type: MarketType = MarketType.MANIFOLD, _place_bet: bool = True
    ) -> None:
        if market_type != MarketType.OMEN:
            raise RuntimeError("Can replicate only into Omen.")

        keys = APIKeys()
        credentials = PrivateCredentials.from_api_keys(keys)
        settings = ReplicateSettings()

        logger.info(
            f"Finalising, resolving and claiming back xDai from existing markets replicated by {credentials.public_key}."
        )
        omen_finalize_and_resolve_and_claim_back_all_markets_based_on_others_tx(
            credentials
        )

        logger.info(
            f"Unfunding soon to be known markets replicated by {credentials.public_key}."
        )
        omen_unfund_replicated_known_markets_tx(
            credentials, saturation_above_threshold=0.9
        )

        logger.info("Redeeming funds from previously unfunded markets.")
        redeem_from_all_user_positions(credentials)

        for close_time_days in settings.CLOSE_TIME_UP_TO_N_DAYS:
            close_time_before = utcnow() + timedelta(days=close_time_days)
            initial_funds_per_market = xdai_type(settings.INITIAL_FUNDS)

            logger.info(
                f"Replicating from {MarketType.MANIFOLD} markets closing in {close_time_days} days."
            )
            omen_replicate_from_tx(
                market_type=MarketType.MANIFOLD,
                n_to_replicate=settings.N_TO_REPLICATE,
                initial_funds=initial_funds_per_market,
                private_credentials=credentials,
                close_time_before=close_time_before,
                auto_deposit=True,
            )
            logger.info(
                f"Replicating from {MarketType.POLYMARKET} markets closing in {close_time_days} days."
            )
            omen_replicate_from_tx(
                market_type=MarketType.POLYMARKET,
                n_to_replicate=settings.N_TO_REPLICATE,
                initial_funds=initial_funds_per_market,
                private_credentials=credentials,
                close_time_before=close_time_before,
                auto_deposit=True,
            )

        logger.debug("Done.")
