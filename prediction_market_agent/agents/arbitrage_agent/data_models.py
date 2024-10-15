import typing as t

from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.tools.betting_strategies.utils import SimpleBet
from pydantic import BaseModel, computed_field


class Correlation(BaseModel):
    near_perfect_correlation: bool | None
    reasoning: str


class ArbitrageBet(BaseModel):
    main_market_bet: SimpleBet
    related_market_bet: SimpleBet


class CorrelatedMarketPair(BaseModel):
    main_market: AgentMarket
    related_market: AgentMarket
    correlation: Correlation

    def __str__(self) -> str:
        return f"main_market {self.main_market.question} related_market_question {self.related_market.question} potential profit {self.potential_profit_per_bet_unit}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def potential_profit_per_bet_unit(self) -> float:
        """
        Calculate potential profit per bet unit based on high positive market correlation.
        For positively correlated markets: Bet YES/NO or NO/YES.
        """

        if self.correlation.near_perfect_correlation is None:
            return 0

        elif self.correlation.near_perfect_correlation > 0:
            min_denominator = min(
                self.main_market.current_p_yes + self.related_market.current_p_no,
                self.main_market.current_p_no + self.related_market.current_p_yes,
            )

        elif self.correlation.near_perfect_correlation < 0:
            min_denominator = min(
                self.main_market.current_p_yes + self.related_market.current_p_yes,
                self.main_market.current_p_no + self.related_market.current_p_no,
            )

        return (1 / min_denominator) - 1

    def bet_directions(self) -> t.Tuple[bool, bool]:
        if self.correlation.near_perfect_correlation is None:
            raise ValueError("Cannot determine bet directions if correlation is None")

        elif self.correlation.near_perfect_correlation > 0:
            # We compare profits for cases YES/NO and NO/YES bets and take the most profitable.
            # For other cases we employ similar logic.
            yes_no = self.main_market.current_p_yes + self.related_market.current_p_no
            no_yes = self.main_market.current_p_no + self.related_market.current_p_yes
            return (True, False) if yes_no >= no_yes else (False, True)

        else:
            yes_yes = self.main_market.current_p_yes + self.related_market.current_p_yes
            no_no = self.main_market.current_p_no + self.related_market.current_p_no
            return (True, True) if yes_yes >= no_no else (False, False)

    def split_bet_amount_between_yes_and_no(
        self, total_bet_amount: float
    ) -> ArbitrageBet:
        """Splits total bet amount following equations below:
        A1/p1 = A2/p2 (same profit regardless of outcome resolution)
        A1 + A2 = total bet amount
        """

        if self.correlation.near_perfect_correlation is None:
            raise ValueError("Cannot split bet amounts when correlation is None")

        bet_direction_main, bet_direction_related = self.bet_directions()

        p_main = (
            self.main_market.current_p_yes
            if bet_direction_main
            else self.main_market.current_p_no
        )
        p_related = (
            self.related_market.current_p_yes
            if bet_direction_related
            else self.related_market.current_p_no
        )
        total_probability = p_main + p_related
        bet_main = total_bet_amount * p_main / total_probability
        bet_related = total_bet_amount * p_related / total_probability
        main_market_bet = SimpleBet(direction=bet_direction_main, size=bet_main)
        related_market_bet = SimpleBet(
            direction=bet_direction_related, size=bet_related
        )
        return ArbitrageBet(
            main_market_bet=main_market_bet, related_market_bet=related_market_bet
        )
