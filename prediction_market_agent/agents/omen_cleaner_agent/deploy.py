from datetime import timedelta

from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.gtypes import HexAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen_resolving import resolve_markets
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import utcnow

from prediction_market_agent.utils import APIKeys

CLEANER_TAG = "cleaner"


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

    @observe()
    def resolve_finalized_markets(self, api_keys: APIKeys) -> list[HexAddress]:
        logger.info("Resolving finalized markets.")
        # Delay by 24 hours to give a chance for the market creator to be resolve it.
        finalized_unresolved_markets = OmenSubgraphHandler().get_omen_binary_markets(
            limit=None, finalized_before=utcnow() - timedelta(hours=24), resolved=False
        )
        return resolve_markets(api_keys, finalized_unresolved_markets)
