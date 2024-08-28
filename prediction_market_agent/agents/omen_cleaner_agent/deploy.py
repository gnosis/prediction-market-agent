from datetime import timedelta

from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.gtypes import (
    HexAddress,
    HexStr,
    IPFSCIDVersion0,
    xdai_type,
)
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import (
    OmenAgentMarket,
    omen_fund_market_tx,
    omen_remove_fund_market_tx,
    redeem_from_all_user_positions,
)
from prediction_market_agent_tooling.markets.omen.omen_resolving import resolve_markets
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import utcnow
from prediction_market_agent_tooling.tools.web3_utils import xdai_to_wei

from prediction_market_agent.agents.replicate_to_omen_agent.image_gen import (
    generate_and_set_image_for_market,
)
from prediction_market_agent.utils import APIKeys

CLEANER_TAG = "cleaner"

# List of markets we know aren't resolvable and it's fine to ignore them.
IGNORED_FOR_RESOLUTION: list[HexAddress] = [
    # Throws `Reverted condition not prepared or found`.
    HexAddress(HexStr("0xf9b90903e98e68a1e69083490ee14c05444a57b4")),
]


class OmenCleanerAgent(DeployableAgent):
    def run(self, market_type: MarketType) -> None:
        if market_type != MarketType.OMEN:
            raise RuntimeError("Can clean only Omen.")

        self.clean()

    @observe()
    def clean(self) -> None:
        self.langfuse_update_current_trace(tags=[CLEANER_TAG])
        api_keys = APIKeys()
        self.resolve_finalized_markets(api_keys)
        self.generate_missing_images(api_keys)
        redeem_from_all_user_positions(api_keys)

    @observe()
    def resolve_finalized_markets(self, api_keys: APIKeys) -> list[HexAddress]:
        logger.info("Resolving finalized markets.")
        # Delay by 24 hours to give a chance for the market creator to be resolve it.
        finalized_unresolved_markets = [
            market
            for market in OmenSubgraphHandler().get_omen_binary_markets(
                limit=None,
                finalized_before=utcnow() - timedelta(hours=24),
                resolved=False,
            )
            if market.id not in IGNORED_FOR_RESOLUTION
        ]
        logger.info(
            f"Found {len(finalized_unresolved_markets)} finalized unresolved markets."
        )
        return resolve_markets(api_keys, finalized_unresolved_markets)

    @observe()
    def generate_missing_images(
        self, api_keys: APIKeys
    ) -> dict[HexAddress, IPFSCIDVersion0 | None]:
        logger.info("Generating missing images.")
        recently_created_markets = OmenSubgraphHandler().get_omen_binary_markets(
            limit=None,
            # Get only serious markets with a reasonable liquidity.
            liquidity_bigger_than=xdai_to_wei(xdai_type(5)),
            # Get only markets created roughly since the last run.
            created_after=utcnow() - timedelta(days=2),
        )
        logger.info(
            f"Found {len(recently_created_markets)} recent markets without an image."
        )
        generated_image_mapping: dict[HexAddress, IPFSCIDVersion0 | None] = {}
        for market in recently_created_markets:
            if OmenSubgraphHandler().get_market_image_url(market.id) is None:
                agent_market = OmenAgentMarket.from_data_model(market)
                # Provide some liquidity to the market to be able to assign the image.
                omen_fund_market_tx(
                    api_keys,
                    agent_market,
                    xdai_to_wei(xdai_type(0.001)),
                    auto_deposit=True,
                )
                generated_image_mapping[market.id] = generate_and_set_image_for_market(
                    market.market_maker_contract_address_checksummed,
                    market.question_title,
                    api_keys,
                )
                # Remove the liquidity we provided, no need to keep it there.
                omen_remove_fund_market_tx(api_keys, agent_market, None)
        return generated_image_mapping
