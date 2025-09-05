import random
from datetime import timedelta

from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.gtypes import (
    USD,
    ChecksumAddress,
    CollateralToken,
    xDai,
)
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen_constants import (
    SDAI_CONTRACT_ADDRESS,
)
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    SAFE_COLLATERAL_TOKENS_ADDRESSES,
)
from prediction_market_agent_tooling.tools.contract_utils import is_erc20_contract
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
from prediction_market_agent.db.replicated_markets_table_handler import (
    ReplicatedMarketsTableHandler,
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
    collateral_token: str | None = None
    initial_funds_in_token: CollateralToken | None = None


class ReplicateSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    REPLICATE: list[ReplicateConfig]
    INITIAL_FUNDS: str

    @property
    def initial_funds_usd(self) -> USD:
        return USD(self.INITIAL_FUNDS)


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
            # Use the initial funds from market, otherwise use the global initial_funds variable.
            initial_funds_per_market: USD | CollateralToken = (
                replicate_config.initial_funds_in_token
                if replicate_config.initial_funds_in_token
                else settings.initial_funds_usd
            )

            collateral_token_address: ChecksumAddress
            if replicate_config.collateral_token:
                collateral_token_address = Web3.to_checksum_address(
                    replicate_config.collateral_token
                )
                # make sure it's ERC20

                if not is_erc20_contract(address=collateral_token_address):
                    raise ValueError(
                        f"Collateral token {collateral_token_address} is not an ERC20."
                    )
            else:
                # Prefer sDai, but create markets in others tokens too.
                collateral_token_address = (
                    SDAI_CONTRACT_ADDRESS
                    if random.random() < 0.8
                    else random.choice(
                        [
                            address
                            for address in SAFE_COLLATERAL_TOKENS_ADDRESSES
                            if address != SDAI_CONTRACT_ADDRESS
                        ]
                    )
                )

            logger.info(
                f"Replicating {replicate_config.n} from {replicate_config.source} markets closing in {replicate_config.close_time_up_to_n_days} days."
            )

            omen_replicate_from_tx(
                market_type=replicate_config.source,
                n_to_replicate=replicate_config.n,
                initial_funds=initial_funds_per_market,
                collateral_token_address=collateral_token_address,
                api_keys=keys,
                close_time_before=close_time_before,
                replicated_market_table_handler=ReplicatedMarketsTableHandler(),
                auto_deposit=True,
            )

            logger.info(f"Replication from {replicate_config.source} done.")

        logger.info("All done.")
