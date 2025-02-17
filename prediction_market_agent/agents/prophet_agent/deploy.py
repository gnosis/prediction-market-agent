from langfuse.openai import AsyncOpenAI
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.deploy.betting_strategy import (
    BettingStrategy,
    KellyBettingStrategy,
)
from prediction_market_agent_tooling.deploy.trade_interval import (
    MarketLifetimeProportionalInterval,
)
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket, SortBy
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.relevant_news_analysis.relevant_news_analysis import (
    get_certified_relevant_news_since_cached,
)
from prediction_market_agent_tooling.tools.relevant_news_analysis.relevant_news_cache import (
    RelevantNewsResponseCache,
)
from prediction_market_agent_tooling.tools.utils import DatetimeUTC, utcnow
from prediction_prophet.benchmark.agents import (
    EmbeddingModel,
    OlasAgent,
    PredictionProphetAgent,
)
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AsyncAnthropic
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

from prediction_market_agent.agents.utils import get_maximum_possible_bet_amount
from prediction_market_agent.utils import DEFAULT_OPENAI_MODEL, APIKeys


class DeployableTraderAgentER(DeployableTraderAgent):
    agent: PredictionProphetAgent | OlasAgent
    bet_on_n_markets_per_run = 3

    @property
    def model(self) -> str | None:
        return self.agent.model

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        prediction = self.agent.predict(market.question)
        logger.info(
            f"Answering '{market.question}' with '{prediction.outcome_prediction}'."
        )
        return prediction.outcome_prediction


class DeployablePredictionProphetGPT4oAgent(DeployableTraderAgentER):
    bet_on_n_markets_per_run = 20
    agent: PredictionProphetAgent

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return KellyBettingStrategy(
            max_bet_amount=get_maximum_possible_bet_amount(
                min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
            ),
            max_price_impact=0.7,
        )

    def load(self) -> None:
        super().load()
        model = "gpt-4o-2024-08-06"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.7),
            ),
            prediction_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.0),
            ),
            include_reasoning=True,
            logger=logger,
        )


class DeployablePredictionProphetGPT4ominiAgent(DeployableTraderAgentER):
    bet_on_n_markets_per_run = 20
    agent: PredictionProphetAgent

    # TODO: Uncomment and configure after we get some historic bet data
    # def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
    #     return KellyBettingStrategy(
    #         max_bet_amount=get_maximum_possible_bet_amount(
    #             min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
    #         ),
    #         max_price_impact=0.7,
    #     )

    def load(self) -> None:
        super().load()
        model = "gpt-4o-mini-2024-07-18"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.7),
            ),
            prediction_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.0),
            ),
            include_reasoning=True,
            logger=logger,
        )


class DeployablePredictionProphetGPT4oAgentNewMarketTrader(
    DeployablePredictionProphetGPT4oAgent
):
    """
    This agent trades on new markets, then re-evaluates positions over each
    market's lifetime, if it observes that news has been published about the
    market since its last trade.
    """

    bet_on_n_markets_per_run = 20
    trade_on_markets_created_after = DatetimeUTC(2024, 10, 31, 0)  # Date of deployment
    get_markets_sort_by = SortBy.NEWEST
    same_market_trade_interval = MarketLifetimeProportionalInterval(max_trades=4)

    def load(self) -> None:
        super().load()
        self.relevant_news_response_cache = RelevantNewsResponseCache()

    def verify_market(self, market_type: MarketType, market: AgentMarket) -> bool:
        if not super().verify_market(market_type, market):
            return False

        # If we have previously traded on this market, check if there is new
        # relevant news that implies we should re-run a full prediction and
        # potentially adjust our position.
        user_id = market.get_user_id(api_keys=APIKeys())
        last_trade_datetime = market.get_most_recent_trade_datetime(user_id=user_id)
        if last_trade_datetime is None:
            return True

        news = get_certified_relevant_news_since_cached(
            question=market.question,
            days_ago=(utcnow() - last_trade_datetime).days,
            cache=self.relevant_news_response_cache,
        )
        return news is not None


class DeployablePredictionProphetGPT4TurboPreviewAgent(DeployableTraderAgentER):
    agent: PredictionProphetAgent

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return KellyBettingStrategy(
            max_bet_amount=get_maximum_possible_bet_amount(
                min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
            ),
            max_price_impact=0.5,
        )

    def load(self) -> None:
        super().load()
        model = "gpt-4-0125-preview"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.7),
            ),
            prediction_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.0),
            ),
            include_reasoning=True,
            logger=logger,
        )


class DeployablePredictionProphetGPT4TurboFinalAgent(DeployableTraderAgentER):
    agent: PredictionProphetAgent

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return KellyBettingStrategy(
            max_bet_amount=get_maximum_possible_bet_amount(
                min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
            ),
            max_price_impact=None,
        )

    def load(self) -> None:
        super().load()
        model = "gpt-4-turbo-2024-04-09"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.7),
            ),
            prediction_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.0),
            ),
            include_reasoning=True,
            logger=logger,
        )


class DeployableOlasEmbeddingOAAgent(DeployableTraderAgentER):
    agent: OlasAgent

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return KellyBettingStrategy(
            max_bet_amount=get_maximum_possible_bet_amount(
                min_=5, max_=25, trading_balance=market.get_trade_balance(APIKeys())
            ),
            max_price_impact=0.5,
        )

    def load(self) -> None:
        super().load()
        model = DEFAULT_OPENAI_MODEL
        api_keys = APIKeys()

        self.agent = OlasAgent(
            research_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.5),
            ),
            prediction_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.0),
            ),
            embedding_model=EmbeddingModel.openai,
            logger=logger,
        )


class DeployablePredictionProphetGPTo1PreviewAgent(DeployableTraderAgentER):
    agent: PredictionProphetAgent
    bet_on_n_markets_per_run = 2

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return KellyBettingStrategy(
            max_bet_amount=get_maximum_possible_bet_amount(
                min_=5, max_=25, trading_balance=market.get_trade_balance(APIKeys())
            ),
            max_price_impact=0.7,
        )

    def load(self) -> None:
        super().load()
        # o1-preview supports only temperature=1.0
        model = "o1-preview-2024-09-12"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=1.0),
            ),
            prediction_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=1.0),
            ),
            include_reasoning=True,
            logger=logger,
        )


class DeployablePredictionProphetGPTo1MiniAgent(DeployableTraderAgentER):
    agent: PredictionProphetAgent

    def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
        return KellyBettingStrategy(
            max_bet_amount=get_maximum_possible_bet_amount(
                min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
            ),
            max_price_impact=None,
        )

    def load(self) -> None:
        super().load()
        # o1-mini supports only temperature=1.0
        model = "o1-mini-2024-09-12"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=1.0),
            ),
            prediction_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=1.0),
            ),
            include_reasoning=True,
            logger=logger,
        )


class DeployablePredictionProphetGPTo1(DeployableTraderAgentER):
    agent: PredictionProphetAgent
    bet_on_n_markets_per_run = 2

    # TODO: Uncomment and configure after we get some historic bet data
    # def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
    #     return KellyBettingStrategy(
    #         max_bet_amount=get_maximum_possible_bet_amount(
    #             min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
    #         ),
    #         max_price_impact=None,
    #     )

    def load(self) -> None:
        super().load()
        # o1 supports only temperature=1.0
        model = "o1-2024-12-17"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=1.0),
            ),
            prediction_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=1.0),
            ),
            include_reasoning=True,
            logger=logger,
        )


class DeployablePredictionProphetGPTo3mini(DeployableTraderAgentER):
    agent: PredictionProphetAgent

    # TODO: Uncomment and configure after we get some historic bet data
    # def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
    #     return KellyBettingStrategy(
    #         max_bet_amount=get_maximum_possible_bet_amount(
    #             min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
    #         ),
    #         max_price_impact=None,
    #     )

    def load(self) -> None:
        super().load()
        # o3-mini supports only temperature=1.0
        model = "o3-mini-2025-01-31"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=1.0),
            ),
            prediction_agent=Agent(
                OpenAIModel(
                    model,
                    openai_client=AsyncOpenAI(
                        api_key=api_keys.openai_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=1.0),
            ),
            include_reasoning=True,
            logger=logger,
        )


class DeployablePredictionProphetClaude3OpusAgent(DeployableTraderAgentER):
    agent: PredictionProphetAgent
    bet_on_n_markets_per_run = 2

    # TODO: Uncomment and configure after we get some historic bet data
    # def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
    #     return KellyBettingStrategy(
    #         max_bet_amount=get_maximum_possible_bet_amount(
    #             min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
    #         ),
    #         max_price_impact=None,
    #     )

    def load(self) -> None:
        super().load()
        # o1-preview supports only temperature=1.0
        model = "claude-3-opus-20240229"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                AnthropicModel(
                    model,
                    anthropic_client=AsyncAnthropic(
                        api_key=api_keys.anthropic_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=1.0),
            ),
            prediction_agent=Agent(
                AnthropicModel(
                    model,
                    anthropic_client=AsyncAnthropic(
                        api_key=api_keys.anthropic_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=1.0),
            ),
            include_reasoning=True,
            logger=logger,
        )


class DeployablePredictionProphetClaude35HaikuAgent(DeployableTraderAgentER):
    bet_on_n_markets_per_run = 20
    agent: PredictionProphetAgent

    # TODO: Uncomment and configure after we get some historic bet data
    # def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
    #     return KellyBettingStrategy(
    #         max_bet_amount=get_maximum_possible_bet_amount(
    #             min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
    #         ),
    #         max_price_impact=None,
    #     )

    def load(self) -> None:
        super().load()
        model = "claude-3-5-haiku-20241022"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                AnthropicModel(
                    model,
                    anthropic_client=AsyncAnthropic(
                        api_key=api_keys.anthropic_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.7),
            ),
            prediction_agent=Agent(
                AnthropicModel(
                    model,
                    anthropic_client=AsyncAnthropic(
                        api_key=api_keys.anthropic_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.0),
            ),
            include_reasoning=True,
            logger=logger,
        )


class DeployablePredictionProphetClaude35SonnetAgent(DeployableTraderAgentER):
    bet_on_n_markets_per_run = 20
    agent: PredictionProphetAgent

    # TODO: Uncomment and configure after we get some historic bet data
    # def get_betting_strategy(self, market: AgentMarket) -> BettingStrategy:
    #     return KellyBettingStrategy(
    #         max_bet_amount=get_maximum_possible_bet_amount(
    #             min_=1, max_=5, trading_balance=market.get_trade_balance(APIKeys())
    #         ),
    #         max_price_impact=None,
    #     )

    def load(self) -> None:
        super().load()
        model = "claude-3-5-sonnet-20241022"
        api_keys = APIKeys()

        self.agent = PredictionProphetAgent(
            research_agent=Agent(
                AnthropicModel(
                    model,
                    anthropic_client=AsyncAnthropic(
                        api_key=api_keys.anthropic_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.7),
            ),
            prediction_agent=Agent(
                AnthropicModel(
                    model,
                    anthropic_client=AsyncAnthropic(
                        api_key=api_keys.anthropic_api_key.get_secret_value()
                    ),
                ),
                model_settings=ModelSettings(temperature=0.0),
            ),
            include_reasoning=True,
            logger=logger,
        )
