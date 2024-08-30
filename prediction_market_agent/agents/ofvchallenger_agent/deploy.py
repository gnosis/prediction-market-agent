from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.langfuse_ import observe

from prediction_market_agent.utils import APIKeys

OFV_CHALLENGER_TAG = "ofv_challenger"


class OFVChallengerAgent(DeployableAgent):
    def run(self, market_type: MarketType) -> None:
        if market_type != MarketType.OMEN:
            raise RuntimeError("Can challenge only Omen.")

        self.challenge()

    @observe()
    def challenge(self) -> None:
        self.langfuse_update_current_trace(tags=[OFV_CHALLENGER_TAG])
        APIKeys()
        # TODO: Coming in the next PR.
