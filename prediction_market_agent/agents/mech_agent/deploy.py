import random
import typing as t

from prediction_market_agent_tooling.benchmark.utils import OutcomePrediction
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket

from prediction_market_agent.tools.mech.utils import (
    MechTool,
    mech_request,
    mech_request_local,
)


class DeployableMechAgentBase(DeployableAgent):
    def __init__(self, tool: MechTool, local: bool):
        self.tool: MechTool = tool
        self.local: bool = local
        super().__init__()

    @property
    def prediction_fn(self) -> t.Callable[[str, MechTool], OutcomePrediction]:
        return mech_request_local if self.local else mech_request

    def pick_markets(self, markets: t.Sequence[AgentMarket]) -> t.Sequence[AgentMarket]:
        # We simply pick 5 random markets to bet on
        markets = list(markets)
        random.shuffle(markets)
        return markets

    def answer_binary_market(self, market: AgentMarket) -> bool:
        result: OutcomePrediction = self.prediction_fn(market.question, self.tool)
        return True if result.p_yes >= 0.5 else False


class DeployableRemoteMechAgentBase(DeployableAgent):
    def __init__(self, tool: MechTool):
        self.tool: MechTool = tool
        super().__init__(tool=tool, local=False)


class DeployablePredictionOnlineAgent(DeployableRemoteMechAgentBase):
    def __init__(self):
        super().__init__(tool=MechTool.PREDICTION_ONLINE)
