from prediction_market_agent_tooling.deploy.agent import (
    Answer,
    BetAmount,
    DeployableTraderAgent,
)
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.manifold.manifold import (
    ManifoldAgentMarket,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.tools.betting_strategies.stretch_bet_between import (
    stretch_bet_between,
)
from prediction_market_agent_tooling.tools.langfuse_ import observe
from prediction_market_agent_tooling.tools.utils import (
    prob_uncertainty,
    should_not_happen,
)
from prediction_prophet.benchmark.agents import (
    EmbeddingModel,
    OlasAgent,
    Prediction,
    PredictionProphetAgent,
    Research,
)

from prediction_market_agent.utils import DEFAULT_OPENAI_MODEL


class PredictionProphetAgentObserved(PredictionProphetAgent):
    @observe()
    def research(
        self, market_question: str, add_langfuse_callback: bool = False
    ) -> Research:
        return super().research(
            market_question=market_question,
            add_langfuse_callback=add_langfuse_callback,
        )

    @observe()
    def make_prediction(
        self,
        market_question: str,
        additional_information: str,
        add_langfuse_callback: bool = False,
    ) -> Prediction:
        return super().make_prediction(
            market_question=market_question,
            additional_information=additional_information,
            add_langfuse_callback=add_langfuse_callback,
        )


class OlasAgentObserved(OlasAgent):
    @observe()
    def research(
        self, market_question: str, add_langfuse_callback: bool = False
    ) -> str:
        return super().research(
            market_question=market_question,
            add_langfuse_callback=add_langfuse_callback,
        )

    @observe()
    def make_prediction(
        self,
        market_question: str,
        additional_information: str,
        add_langfuse_callback: bool = False,
    ) -> Prediction:
        return super().make_prediction(
            market_question=market_question,
            additional_information=additional_information,
            add_langfuse_callback=add_langfuse_callback,
        )


class DeployableTraderAgentER(DeployableTraderAgent):
    agent: PredictionProphetAgentObserved | OlasAgentObserved
    bet_on_n_markets_per_run = 1

    @property
    def model(self) -> str | None:
        return self.agent.model

    def calculate_bet_amount(self, answer: Answer, market: AgentMarket) -> BetAmount:
        amount: float
        max_bet_amount: float
        if isinstance(market, ManifoldAgentMarket):
            # Manifold won't give us fractional Mana, so bet the minimum amount to win at least 1 Mana.
            amount = market.get_minimum_bet_to_win(answer.decision, amount_to_win=1)
            max_bet_amount = 10.0
        elif isinstance(market, OmenAgentMarket):
            # TODO: After https://github.com/gnosis/prediction-market-agent-tooling/issues/161 is done,
            # use agent's probability to calculate the amount.
            market_liquidity = market.get_liquidity_in_xdai()
            amount = stretch_bet_between(
                Probability(
                    prob_uncertainty(market.current_p_yes)
                ),  # Not a probability, but it's a value between 0 and 1, so it's fine.
                min_bet=0.5,
                max_bet=1.0,
            )
            if answer.decision == (market.current_p_yes > 0.5):
                amount = amount * 0.75
            else:
                amount = amount * 1.25
            max_bet_amount = (
                2.0 if market_liquidity > 5 else 0.1 if market_liquidity > 1 else 0
            )
        else:
            should_not_happen(f"Unknown market type: {market}")
        if amount > max_bet_amount:
            logger.warning(
                f"Calculated amount {amount} {market.currency} is exceeding our limit {max_bet_amount=}, betting only {market.get_tiny_bet_amount()} for benchmark purposes."
            )
            amount = market.get_tiny_bet_amount().amount
        return BetAmount(amount=amount, currency=market.currency)

    def answer_binary_market(self, market: AgentMarket) -> Answer | None:
        prediciton = self.agent.predict(
            market.question,
            add_langfuse_callback=self.enable_langfuse,
        )
        if prediciton.outcome_prediction is None:
            logger.error(f"Prediction failed for {market.question}.")
            return None
        logger.info(
            f"Answering '{market.question}' with '{prediciton.outcome_prediction.decision}'."
        )
        return prediciton.outcome_prediction


class DeployablePredictionProphetGPT4oAgent(DeployableTraderAgentER):
    agent = PredictionProphetAgentObserved(model="gpt-4o-2024-08-06")


class DeployablePredictionProphetGPT4TurboPreviewAgent(DeployableTraderAgentER):
    agent = PredictionProphetAgentObserved(model="gpt-4-0125-preview")


class DeployablePredictionProphetGPT4TurboFinalAgent(DeployableTraderAgentER):
    agent = PredictionProphetAgentObserved(model="gpt-4-turbo-2024-04-09")


class DeployableOlasEmbeddingOAAgent(DeployableTraderAgentER):
    agent = OlasAgentObserved(
        model=DEFAULT_OPENAI_MODEL, embedding_model=EmbeddingModel.openai
    )
