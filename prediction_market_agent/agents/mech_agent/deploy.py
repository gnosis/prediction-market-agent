import random
import typing as t

from prediction_market_agent_tooling.benchmark.utils import OutcomePrediction
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket

from prediction_market_agent.tools.mech.utils import (
    MechResponse,
    MechTool,
    mech_request,
    mech_request_local,
)


class DeployableMechAgentBase(DeployableAgent):
    bet_on_n_markets_per_run: int = 5

    def load(self) -> None:
        self.tool: MechTool | None = None
        self.local: bool | None = None

    @property
    def prediction_fn(self) -> t.Callable[[str, MechTool], MechResponse]:
        if self.local is None:
            raise ValueError("Local mode not set")

        return mech_request_local if self.local else mech_request

    def answer_binary_market(self, market: AgentMarket) -> OutcomePrediction:
        if self.tool is None:
            raise ValueError("Tool not set")

        response: MechResponse = self.prediction_fn(market.question, self.tool)
        outcome_prediction = OutcomePrediction(
            decision=response.p_yes > 0.5,
            p_yes=response.p_yes,
            confidence=response.confidence,
            info_utility=response.info_utility,
        )
        return outcome_prediction


class DeployablePredictionOnlineAgent(DeployableMechAgentBase):
    def load(self) -> None:
        self.local = True
        self.tool = MechTool.PREDICTION_ONLINE


class DeployablePredictionOfflineAgent(DeployableMechAgentBase):
    def load(self) -> None:
        self.local = True
        self.tool = MechTool.PREDICTION_OFFLINE


class DeployablePredictionOnlineSMEAgent(DeployableMechAgentBase):
    def load(self) -> None:
        self.local = True
        self.tool = MechTool.PREDICTION_ONLINE_SME


class DeployablePredictionOfflineSMEAgent(DeployableMechAgentBase):
    def load(self) -> None:
        self.local = True
        self.tool = MechTool.PREDICTION_OFFLINE_SME


class DeployablePredictionRequestRAGAgent(DeployableMechAgentBase):
    def load(self) -> None:
        self.local = True
        self.tool = MechTool.PREDICTION_REQUEST_RAG


class DeployablePredictionRequestReasoningAgent(DeployableMechAgentBase):
    def load(self) -> None:
        self.local = True
        self.tool = MechTool.PREDICTION_REQUEST_REASONING


class DeployablePredictionUrlCotAgent(DeployableMechAgentBase):
    def load(self) -> None:
        self.local = True
        self.tool = MechTool.PREDICTION_URL_COT


class DeployablePredictionWithResearchBoldAgent(DeployableMechAgentBase):
    def load(self) -> None:
        self.local = True
        self.tool = MechTool.PREDICTION_WITH_RESEARCH_REPORT_BOLD
