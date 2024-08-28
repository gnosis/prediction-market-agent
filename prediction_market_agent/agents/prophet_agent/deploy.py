from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import KellyBettingStrategy
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.tools.tavily_storage.tavily_models import (
    TavilyStorage,
)
from prediction_prophet.benchmark.agents import (
    EmbeddingModel,
    OlasAgent,
    PredictionProphetAgent,
)

from prediction_market_agent.utils import DEFAULT_OPENAI_MODEL


class DeployableTraderAgentER(DeployableTraderAgent):
    agent: PredictionProphetAgent | OlasAgent
    bet_on_n_markets_per_run = 1

    @property
    def model(self) -> str | None:
        return self.agent.model

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        prediction = self.agent.predict(market.question)
        if prediction.outcome_prediction is None:
            logger.error(f"Prediction failed for {market.question}.")
            return None
        logger.info(
            f"Answering '{market.question}' with probability '{prediction.outcome_prediction.p_yes}'."
        )
        return prediction.outcome_prediction


class DeployablePredictionProphetGPT4oAgent(DeployableTraderAgentER):
    agent: PredictionProphetAgent

    def load(self) -> None:
        super().load()
        self.agent = PredictionProphetAgent(
            model="gpt-4o-2024-08-06",
            tavily_storage=TavilyStorage(agent_id=self.__class__.__name__),
            logger=logger,
        )


class DeployablePredictionProphetGPT4TurboPreviewAgent(DeployableTraderAgentER):
    agent: PredictionProphetAgent

    def load(self) -> None:
        super().load()
        self.agent = PredictionProphetAgent(
            model="gpt-4-0125-preview",
            tavily_storage=TavilyStorage(agent_id=self.__class__.__name__),
            logger=logger,
        )


class DeployablePredictionProphetGPT4TurboFinalAgent(DeployableTraderAgentER):
    agent: PredictionProphetAgent

    def load(self) -> None:
        super().load()
        self.agent = PredictionProphetAgent(
            model="gpt-4-turbo-2024-04-09",
            tavily_storage=TavilyStorage(agent_id=self.__class__.__name__),
            logger=logger,
        )


class DeployablePredictionProphetGPT4KellyAgent(
    DeployablePredictionProphetGPT4TurboFinalAgent
):
    strategy = KellyBettingStrategy()


class DeployableOlasEmbeddingOAAgent(DeployableTraderAgentER):
    agent: OlasAgent

    def load(self) -> None:
        super().load()
        self.agent = OlasAgent(
            model=DEFAULT_OPENAI_MODEL, embedding_model=EmbeddingModel.openai
        )
