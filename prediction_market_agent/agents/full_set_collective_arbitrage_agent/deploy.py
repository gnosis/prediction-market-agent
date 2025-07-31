import math
from enum import Enum

from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.gtypes import (
    CollateralToken,
    OutcomeStr,
    OutcomeToken,
    Probability,
)
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    ProcessedTradedMarket,
)
from prediction_market_agent_tooling.markets.data_models import (
    USD,
    CategoricalProbabilisticAnswer,
    PlacedTrade,
    TradeType,
)
from prediction_market_agent_tooling.markets.market_fees import MarketFees
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.seer.seer import SeerAgentMarket
from prediction_market_agent_tooling.markets.seer.seer_contracts import (
    SwaprRouterContract,
)
from prediction_market_agent_tooling.tools.tokens.auto_deposit import (
    mint_full_set_for_market,
)
from web3 import Web3


class ArbitrageType(Enum):
    UNDER_ESTIMATED = "underestimated"
    OVER_ESTIMATED = "overestimated"


class CalculationType(Enum):
    COST = "cost"
    REVENUE = "revenue"


class DeployableFullSetCollectiveArbitrageAgent(DeployableTraderAgent):
    supported_markets = [MarketType.SEER]

    def load(self) -> None:
        super().load()
        self.epsilon = 0.02  # 2% error margin for probabilities as rounding error
        self.swapr_router = SwaprRouterContract()
        self.api_keys = APIKeys()
        self.max_mint_amount_dollars: float = (
            0.5  # Minting is done per 0.5$ outcome set
        )
        self.max_merge_amount_dollars: float = (
            0.5  # Buying is done per 0.5$ outcome set
        )

    def run(self, market_type: MarketType) -> None:
        if market_type not in self.supported_markets:
            raise RuntimeError(
                f"Can arbitrage only on {self.supported_markets} markets."
            )
        super().run(market_type=market_type)

    def before_process_market(
        self, market_type: MarketType, market: AgentMarket
    ) -> None:
        super().before_process_market(market_type, market)
        self.check_min_required_balance_to_trade(market)

        if not isinstance(market, SeerAgentMarket):
            raise ValueError(
                f"This agent only supports SeerAgentMarket, got {type(market)}"
            )

        # Check if we have complete outcome token pool data
        if not market.outcome_token_pool:
            logger.info(
                f"Market '{market.question}' has no outcome token pool data - skipping"
            )
            return None

        # Multiresult markets are not supported
        if market.is_multiresult:
            logger.info(
                f"Market '{market.question}' is a multiresult market - skipping"
            )
            return None

        # Check if all outcomes are present in the outcome token pool
        for outcome in market.outcomes:
            if outcome not in market.outcome_token_pool:
                logger.info(
                    f"Market '{market.question}' missing outcome '{outcome}' in token pool - skipping market"
                )
                return None

    def process_market(
        self,
        market_type: MarketType,
        market: AgentMarket,
        verify_market: bool = True,
    ) -> ProcessedTradedMarket | None:
        if not isinstance(market, SeerAgentMarket):
            raise ValueError(
                f"This agent only supports SeerAgentMarket, got {type(market)}"
            )

        prob_summed = sum(market.probabilities.values())
        overestimated_prob = prob_summed > 1 + self.epsilon
        underestimated_prob = prob_summed < 1 - self.epsilon

        if overestimated_prob or underestimated_prob:
            logger.info(
                f"Market '{market.question}' has probabilities that don't sum to 1. Sum: {prob_summed}"
            )

            try:
                placed_trades: list[PlacedTrade] = []
                arbitrage_type: ArbitrageType | None = None

                if underestimated_prob:
                    logger.info(
                        "Under-estimated probabilities detected - buy low, sell high arbitrage"
                    )
                    arbitrage_type = ArbitrageType.UNDER_ESTIMATED
                    trades = self.merge(market)
                    placed_trades.extend(trades)

                if overestimated_prob:
                    logger.info(
                        "Over-estimated probabilities detected - mint and sell arbitrage"
                    )
                    arbitrage_type = ArbitrageType.OVER_ESTIMATED
                    # For over-estimated: mint complete sets (cost 1.0) then sell (get > 1.0)
                    sell_trades = self._arbitrage_overestimated(market)
                    placed_trades.extend(sell_trades)

                return ProcessedTradedMarket(
                    answer=CategoricalProbabilisticAnswer(
                        probabilities=market.probabilities,
                        confidence=1.0,
                        reasoning=f"Arbitrage: {arbitrage_type.value if arbitrage_type else 'unknown'} probabilities (sum: {prob_summed:.4f}). Executed {len(placed_trades)} trades.",
                    ),
                    trades=placed_trades,
                )

            except Exception as e:
                logger.error(f"Market '{market.question}' - skipping due to error {e}")
                return None

        return None

    def merge(self, market: SeerAgentMarket) -> list[PlacedTrade]:
        placed_trades: list[PlacedTrade] = []
        max_sets_to_arbitrage = self._max_sets(market, CalculationType.COST)
        max_sets_to_arbitrage = min(
            max_sets_to_arbitrage, self.max_merge_amount_dollars
        )

        if max_sets_to_arbitrage > 0:
            placed_trades.extend(
                self._trade_complete_sets(market, max_sets_to_arbitrage, TradeType.BUY)
            )
            if len(placed_trades) > 0:
                placed_trades.extend(
                    self._trade_complete_sets(
                        market, max_sets_to_arbitrage, TradeType.SELL
                    )
                )

        return placed_trades

    def _arbitrage_overestimated(self, market: SeerAgentMarket) -> list[PlacedTrade]:
        logger.info(
            "Executing mint-and-sell arbitrage for over-estimated probabilities"
        )

        placed_trades: list[PlacedTrade] = []

        try:
            optimal_sets = self._max_sets(market, CalculationType.REVENUE)
            # Minting is done per 1$ per set
            optimal_sets = min(float(optimal_sets), self.max_mint_amount_dollars)

            if optimal_sets <= 0:
                logger.info("No profitable minting opportunity found")
                return placed_trades

            available_balance = market.get_trade_balance(self.api_keys)
            required_balance = market.get_in_usd(CollateralToken(optimal_sets))

            if available_balance < required_balance:
                logger.warning(
                    f"Insufficient balance: need {required_balance}, have {available_balance}"
                )
                # Scale down to available balance
                affordable_sets = int(available_balance.value)
                if affordable_sets <= 0:
                    return placed_trades
                mint_amount = OutcomeToken(affordable_sets)
                logger.info(
                    f"Scaling down to affordable amount: {affordable_sets} sets"
                )
            else:
                mint_amount = OutcomeToken(optimal_sets)

            logger.info(f"Minting {mint_amount.value} complete sets for arbitrage")
            collateral_token_address = Web3.to_checksum_address(
                market.get_collateral_token_contract().address
            )
            mint_full_set_for_market(
                market_collateral_token=collateral_token_address,
                market_id=Web3.to_checksum_address(market.id),
                collateral_amount_wei=mint_amount.as_outcome_wei.as_wei,
                api_keys=self.api_keys,
                web3=None,
            )
            # Tokens will be minted by Autodeposit inside of place_bet method in seer.py
            placed_trades.extend(
                self._trade_complete_sets(market, mint_amount.value, TradeType.SELL)
            )

            return placed_trades

        except Exception as e:
            logger.error(f"Over-estimated arbitrage failed: {e}")
            raise ValueError(f"Failed to execute over-estimated arbitrage: {e}") from e

        return placed_trades

    def _max_sets(
        self,
        market: SeerAgentMarket,
        calc_type: CalculationType,
        target: float = 1.0,
        max_expand: float = 1e6,
    ) -> float:
        """
        Largest integer n such that the *marginal* quote is still profitable vs `target` USD.

        COST    -> marginal_cost(n)  < target
        REVENUE -> marginal_revenue(n) > target
        """

        pools = (
            {o: t.value for o, t in market.outcome_token_pool.items() if t.value > 0}
            if market.outcome_token_pool
            else {}
        )
        probs = (
            {o: float(p) for o, p in market.probabilities.items() if p > 0}
            if market.probabilities
            else {}
        )
        k = {o: probs[o] * pools[o] ** 2 for o in pools}

        if (
            len(pools) == 0
            or len(probs) == 0
            or sum(probs.values()) == 0
            or sum(pools.values()) == 0
        ):
            return 0

        mq0 = self._get_marginal_quote(0.0, k, pools, market.fees, market, calc_type)

        if calc_type == CalculationType.COST:
            # Need marginal cost < 1 to be worth buying
            if mq0 >= target:
                return 0
            low = 0.0
            # You cannot buy more than the smallest pool has (denominator would hit zero)
            high = min(pools.values()) - 1e-12
            good = lambda q: q < target
        else:  # REVENUE
            # Need marginal revenue > 1 to be worth minting/selling
            if mq0 <= target:
                return 0
            low, high = 0.0, 1.0
            while (
                self._get_marginal_quote(high, k, pools, market.fees, market, calc_type)
                > target
                and high < max_expand
            ):
                low, high = high, high * 2.0
            good = lambda q: q > target

        # --- Binary search ---
        for _ in range(64):
            mid = (low + high) / 2.0
            if good(
                self._get_marginal_quote(mid, k, pools, market.fees, market, calc_type)
            ):
                low = mid
            else:
                high = mid

        return max(int(math.floor(low + self.epsilon)), 0)

    def _get_marginal_quote(
        self,
        n: float,
        k: dict[OutcomeStr, float],
        pools: dict[OutcomeStr, float],
        fees: MarketFees,
        market: SeerAgentMarket,
        calc_type: CalculationType,
    ) -> float:
        """
        Return the *marginal* USD value of trading one more complete set at depth n.
        COST  -> how much you must pay
        REVENUE -> how much you will receive
        """

        if calc_type == CalculationType.COST:
            if any(n >= pools[o] for o in pools):
                return float("inf")
            gross = sum(k[o] / (pools[o] - n) ** 2 for o in pools)

            if fees.bet_proportion > 0:
                gross = gross / (1 - fees.bet_proportion)
            return gross + fees.absolute

        else:
            gross = sum(k[o] / (pools[o] + n) ** 2 for o in pools)
            if fees.bet_proportion > 0:
                gross = gross * (1 - fees.bet_proportion)
            return max(gross - fees.absolute, 0.0)

    def _trade_complete_sets(
        self, market: SeerAgentMarket, quantity: float, trade_type: TradeType
    ) -> list[PlacedTrade]:
        """
        Execute trades for complete sets of outcome tokens.

        Args:
            market: The market to trade on
            quantity: Number of complete sets to trade
            trade_type: BUY or SELL
        """
        action = "Buying" if trade_type == TradeType.BUY else "Selling"
        logger.info(f"{action} {quantity} complete sets for market {market.question}")

        placed_trades: list[PlacedTrade] = []
        lost_amount = USD(0)
        try:
            for outcome in market.outcomes:
                amount_usd = None
                try:
                    if market.probabilities[outcome] == Probability(0):
                        continue

                    if trade_type == TradeType.BUY:
                        logger.info(
                            f"Buying {quantity} complete sets for market {market.question}"
                        )
                        amount_usd = USD(quantity)
                        placed_trade = PlacedTrade(
                            trade_type=TradeType.BUY,
                            outcome=outcome,
                            amount=amount_usd,
                            id=market.buy_tokens(outcome=outcome, amount=amount_usd),
                        )
                    else:
                        user_id = market.get_user_id(self.api_keys)
                        current_position = market.get_position(user_id)

                        tokens_to_sell = (
                            min(
                                OutcomeToken(quantity),
                                current_position.amounts_ot[outcome],
                            )
                            if current_position
                            else OutcomeToken(0)
                        )
                        amount_usd = market.get_in_usd(
                            market.get_sell_value_of_outcome_token(
                                outcome, tokens_to_sell
                            )
                        )
                        logger.info(
                            f"Selling {tokens_to_sell.value} {outcome} for {amount_usd.value} USD"
                        )
                        placed_trade = PlacedTrade(
                            trade_type=TradeType.SELL,
                            outcome=outcome,
                            amount=amount_usd,
                            id=market.sell_tokens(
                                outcome=outcome, amount=tokens_to_sell
                            ),
                        )

                    placed_trades.append(placed_trade)
                    logger.info(f"Trade executed with ID: {placed_trade.id}")
                except Exception as e:
                    trade_loss = amount_usd if amount_usd else USD(0)
                    lost_amount += trade_loss
                    logger.error(
                        f"Failed to execute trade for outcome {outcome}: {e} with amount {trade_loss.value} USD"
                    )

            logger.info(
                f"Placed {len(placed_trades)} trades for market {market.question} for total of {sum(trade.amount.value for trade in placed_trades)} USD. Lost {lost_amount.value} USD"
            )

        except Exception as e:
            action_lower = action.lower()
            logger.error(f"Failed to {action_lower} complete sets: {e}")
            raise ValueError(
                f"Failed to execute complete set {action_lower}: {e}"
            ) from e

        return placed_trades
