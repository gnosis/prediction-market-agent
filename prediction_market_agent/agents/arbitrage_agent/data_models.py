import typing as t

from prediction_market_agent_tooling.gtypes import USD, OutcomeStr
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import Trade
from prediction_market_agent_tooling.tools.utils import check_not_none
from pydantic import BaseModel


class Bet(BaseModel):
    direction: OutcomeStr
    size: USD


class Correlation(BaseModel):
    near_perfect_correlation: bool | None
    reasoning: str


class ArbitrageBet(BaseModel):
    main_market_bet: Bet
    related_market_bet: Bet


class CorrelatedMarketPair(BaseModel):
    main_market: AgentMarket
    related_market: AgentMarket
    correlation: Correlation

    def __str__(self) -> str:
        return f"main_market {self.main_market.question} related_market_question {self.related_market.question} potential_profit_per_unit {self.potential_profit_per_bet_unit()}"

    @property
    def main_market_and_related_market_equal(self) -> bool:
        return self.main_market.id.lower() == self.related_market.id.lower()

    def potential_profit_per_bet_unit(self) -> float:
        """
        Calculate potential profit per bet unit based on high positive market correlation.
        For positively correlated markets: Bet YES/NO or NO/YES.
        """

        if self.correlation.near_perfect_correlation is None:
            return 0

        bet_direction_main, bet_direction_related = self.bet_directions()
        p_main = self.main_market.p_yes if bet_direction_main else self.main_market.p_no
        p_related = (
            self.related_market.p_yes
            if bet_direction_related
            else self.related_market.p_no
        )
        denominator = p_main + p_related
        return (1 / denominator) - 1

    def bet_directions(self) -> t.Tuple[bool, bool]:
        correlation = check_not_none(self.correlation.near_perfect_correlation)
        if correlation:
            # We compare denominators for cases YES/NO and NO/YES bets and take the most profitable (i.e. where denominator is the lowest).
            # For other cases we employ similar logic.
            yes_no = self.main_market.p_yes + self.related_market.p_no
            no_yes = self.main_market.p_no + self.related_market.p_yes
            return (True, False) if yes_no <= no_yes else (False, True)

        else:
            yes_yes = self.main_market.p_yes + self.related_market.p_yes
            no_no = self.main_market.p_no + self.related_market.p_no
            return (True, True) if yes_yes <= no_no else (False, False)

    def split_bet_amount_between_yes_and_no(
        self, total_bet_amount: USD
    ) -> ArbitrageBet:
        """Splits total bet amount following equations below:
        A1/p1 = A2/p2 (same profit regardless of outcome resolution)
        A1 + A2 = total bet amount
        """

        bet_direction_main, bet_direction_related = self.bet_directions()

        p_main = self.main_market.p_yes if bet_direction_main else self.main_market.p_no
        p_related = (
            self.related_market.p_yes
            if bet_direction_related
            else self.related_market.p_no
        )
        total_probability = p_main + p_related
        bet_main = total_bet_amount * p_main / total_probability
        bet_related = total_bet_amount * p_related / total_probability
        bet_direction_main_outcome = self.main_market.get_outcome_str_from_bool(
            bet_direction_main
        )
        main_market_bet = Bet(direction=bet_direction_main_outcome, size=bet_main)
        bet_direction_related_outcome = self.related_market.get_outcome_str_from_bool(
            bet_direction_related
        )
        related_market_bet = Bet(
            direction=bet_direction_related_outcome,
            size=bet_related,
        )
        return ArbitrageBet(
            main_market_bet=main_market_bet, related_market_bet=related_market_bet
        )


class MarketTrade(Trade):
    market: AgentMarket
