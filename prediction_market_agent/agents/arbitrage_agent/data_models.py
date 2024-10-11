import typing as t

from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from pydantic import BaseModel, computed_field


class Correlation(BaseModel):
    near_perfect_correlation: bool
    reasoning: str


class CorrelatedMarketPair(BaseModel):
    main_market: AgentMarket
    related_market: AgentMarket

    def __repr__(self) -> str:
        return f"main_market {self.main_market.question} related_market_question {self.related_market.question} potential profit {self.potential_profit_per_bet_unit}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def potential_profit_per_bet_unit(self) -> float:
        """
        Calculate potential profit per bet unit based on high positive market correlation.
        For positively correlated markets: Bet YES/NO or NO/YES.
        """
        # Smaller correlations will be handled in a future ticket
        # https://github.com/gnosis/prediction-market-agent/issues/508
        # Negative correlations are not yet supported by the current LLM prompt, hence not handling those for now.
        p_yes = min(self.main_market.current_p_yes, self.related_market.current_p_yes)
        p_no = min(self.main_market.current_p_no, self.related_market.current_p_no)
        total_probability = p_yes + p_no

        # Ensure total_probability is non-zero to avoid division errors
        if total_probability > 0:
            return (1.0 / total_probability) - 1.0
        else:
            return 0  # No arbitrage possible if the sum of probabilities is zero

    @computed_field  # type: ignore[prop-decorator]
    @property
    def market_to_bet_yes(self) -> AgentMarket:
        return (
            self.main_market
            if self.main_market.current_p_yes <= self.related_market.current_p_yes
            else self.related_market
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def market_to_bet_no(self) -> AgentMarket:
        return (
            self.main_market
            if self.main_market.current_p_yes > self.related_market.current_p_yes
            else self.related_market
        )

    def split_bet_amount_between_yes_and_no(
        self, total_bet_amount: float
    ) -> t.Tuple[float, float]:
        """Splits total bet amount following equations below:
        A1/p1 = A2/p2 (same profit regardless of outcome resolution)
        A1 + A2 = total bet amount
        """
        amount_to_bet_yes = (
            total_bet_amount
            * self.market_to_bet_yes.current_p_yes
            / (
                self.market_to_bet_yes.current_p_yes
                + self.market_to_bet_no.current_p_no
            )
        )
        amount_to_bet_no = total_bet_amount - amount_to_bet_yes
        return amount_to_bet_yes, amount_to_bet_no
