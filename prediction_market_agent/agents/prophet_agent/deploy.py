from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    MaxAccuracyBettingStrategy,
)
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
    strategy = MaxAccuracyBettingStrategy(bet_amount=1)

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
    bet_on_n_markets_per_run = 20
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


class DeployableOlasEmbeddingOAAgent(DeployableTraderAgentER):
    agent: OlasAgent
    bet_on_n_markets_per_run = 20

    def load(self) -> None:
        super().load()
        self.agent = OlasAgent(
            model=DEFAULT_OPENAI_MODEL, embedding_model=EmbeddingModel.openai
        )


class DeployablePredictionProphetGPTo1PreviewAgent(DeployableTraderAgentER):
    agent: PredictionProphetAgent

    def load(self) -> None:
        super().load()
        # o1-preview supports only temperature=1.0
        self.agent = PredictionProphetAgent(
            model="o1-preview-2024-09-12",
            research_temperature=1.0,
            prediction_temperature=1.0,
            tavily_storage=TavilyStorage(agent_id=self.__class__.__name__),
            logger=logger,
        )


class DeployablePredictionProphetGPTo1MiniAgent(DeployableTraderAgentER):
    bet_on_n_markets_per_run = 20
    agent: PredictionProphetAgent

    def load(self) -> None:
        super().load()
        # o1-mini supports only temperature=1.0
        self.agent = PredictionProphetAgent(
            model="o1-mini-2024-09-12",
            research_temperature=1.0,
            prediction_temperature=1.0,
            tavily_storage=TavilyStorage(agent_id=self.__class__.__name__),
            logger=logger,
        )
