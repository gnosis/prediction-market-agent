import typing as t
import hashlib
from enum import Enum
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.agent_market import AgentMarket, ProcessedTradedMarket
from prediction_market_agent_tooling.gtypes import OutcomeStr, Probability
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer, CategoricalProbabilisticAnswer, Trade, PlacedTrade, TradeType, USD
from prediction_market_agent_tooling.tools.tokens.slippage import get_slippage_tolerance_per_token
from prediction_market_agent_tooling.gtypes import CollateralToken, OutcomeToken, Wei, ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.markets.seer.seer import SeerAgentMarket
from prediction_market_agent_tooling.markets.seer.seer_contracts import SwaprRouterContract
from prediction_market_agent_tooling.markets.seer.data_models import ExactInputSingleParams
from prediction_market_agent_tooling.tools.contract import ContractERC20OnGnosisChain
from web3 import Web3
import math
from prediction_market_agent_tooling.markets.market_fees import MarketFees


class ArbitrageType(Enum):
    UNDER_ESTIMATED = "underestimated"
    OVER_ESTIMATED = "overestimated"


class DeployableFullSetCollectiveArbitrageAgent(DeployableTraderAgent):
    supported_markets = [MarketType.SEER]

    def load(self) -> None:
        super().load()
        self.epsilon = 0.015  # 1.5% error margin for probabilities as rounding error
        self.swapr_router = SwaprRouterContract()
        self.api_keys = APIKeys()

        # Expose slippage curvature (alpha) so it can be tuned per deployment.
        # 0.8 matches our current empirical estimate but is **not** hard-coded in
        # the pricing logic – override `self.slippage_alpha` for different pools.
        self.slippage_alpha: float = 0.8

    def run(self, market_type: MarketType) -> None:
        if market_type not in self.supported_markets:
            raise RuntimeError(
                f"Can arbitrage only on {self.supported_markets} markets."
            )
        super().run(market_type=market_type)

    def get_markets(
        self,
        market_type: MarketType,
    ) -> t.Sequence[AgentMarket]:

        # Fetch all markets to choose from
        available_markets = SeerAgentMarket.get_markets(
            limit=self.n_markets_to_fetch,
            sort_by=self.get_markets_sort_by,
            filter_by=self.get_markets_filter_by,
            created_after=self.trade_on_markets_created_after,
            include_conditional_markets=True,
        )
        return available_markets

    def process_market(
        self,
        market_type: MarketType,
        market: AgentMarket,
        verify_market: bool = True,
    ) -> ProcessedTradedMarket | None:
        if verify_market and not self.verify_market(market_type, market):
            logger.info(f"Market '{market.question}' doesn't meet the criteria.")
            return None
        
        # Check if we have complete outcome token pool data
        if not market.outcome_token_pool:
            logger.info(f"Market '{market.question}' has no outcome token pool data - skipping")
            return None
            
        # Check if all outcomes are present in the outcome token pool
        for outcome in market.outcomes:
            if outcome not in market.outcome_token_pool:
                logger.info(f"Market '{market.question}' missing outcome '{outcome}' in token pool - skipping market")
                return None
            
        prob_summed = sum(market.probabilities.values())
        overestimated_prob = prob_summed > 1 + self.epsilon
        underestimated_prob = prob_summed < 1 - self.epsilon

        if overestimated_prob or underestimated_prob:
            logger.info(f"Market '{market.question}' has probabilities that don't sum to 1. Sum: {prob_summed}")

            try:
                placed_trades: list[PlacedTrade] = []
                arbitrage_type: ArbitrageType | None = None
                
                if underestimated_prob:
                    logger.info("Under-estimated probabilities detected - buy low, sell high arbitrage")
                    arbitrage_type = ArbitrageType.UNDER_ESTIMATED
                    max_sets_to_arbitrage = self.max_sets_to_buy(market)
                    if max_sets_to_arbitrage > 0:
                        # Buy underpriced outcome tokens
                        buy_trades = self._buy_complete_sets(market, max_sets_to_arbitrage)
                        if buy_trades:
                            placed_trades.extend(buy_trades)
                            
                            # Immediately sell those tokens to realize profit
                            sell_trades = self._sell_complete_sets(market, max_sets_to_arbitrage)
                            placed_trades.extend(sell_trades)

                if overestimated_prob:
                    logger.info("Over-estimated probabilities detected - mint and sell arbitrage")
                    arbitrage_type = ArbitrageType.OVER_ESTIMATED
                    # For over-estimated: mint complete sets (cost 1.0) then sell (get > 1.0)
                    sell_trades = self._arbitrage_overestimated(market)
                    placed_trades.extend(sell_trades)
                
    
                return ProcessedTradedMarket(
                    answer=CategoricalProbabilisticAnswer(
                        probabilities=market.probabilities,
                        confidence=1.0,
                        reasoning=f"Arbitrage: {arbitrage_type.value if arbitrage_type else 'unknown'} probabilities (sum: {prob_summed:.4f}). Executed {len(placed_trades)} trades."
                    ),
                    trades=placed_trades
                )
                
            except Exception as e:
                logger.error(f"Market '{market.question}' - skipping due to error {e}")
                return None
        
        return None

    def estimate_arbitrage_scale(self, market: AgentMarket) -> float:
        liquidity_usd = market.get_token_in_usd(market.get_liquidity())
        arbitrage_gap = abs(1.0 - sum(market.probabilities.values()))
        arbitrage_gap_scaled=(arbitrage_gap * liquidity_usd.value) * (arbitrage_gap / (arbitrage_gap + self.epsilon))

        return 10 ** math.floor(math.log10(arbitrage_gap_scaled))

    def _calculate_cost_to_buy_token(
        self,
        *,
        pool_size: float,
        probability: float,
        tokens_to_buy: float,
        fees: MarketFees | None = None,
        alpha: float | None = None,
    ) -> float | None:
        """
        Estimate collateral (USD-denominated) needed to purchase `tokens_to_buy`
        from a pool with `pool_size` tokens priced at `probability`.

        Uses the same Uniswap V3-style curve as the test helper:
            tokens_out = P * (1 - (P / (P + amount / prob)) ** alpha)

        The formula is inverted to solve for `amount`. Returns `None` when the
        """
        # Use provided alpha or fall back to the agent-level default.
        alpha = alpha if alpha is not None else self.slippage_alpha

        # Normalise fees – treat missing fees as zeroes.
        fees = fees or MarketFees.get_zero_fees()

        if tokens_to_buy <= 0 or tokens_to_buy >= 0.95 * pool_size:
            return None

        # Invert the pricing curve used by Uniswap-V3-style pools.
        denominator = (1.0 - tokens_to_buy / pool_size) ** (1.0 / alpha)
        amount_before_fee = probability * (pool_size / denominator - pool_size)
        if amount_before_fee <= 0:
            return None


        if fees.bet_proportion > 0:
            amount_before_fee /= 1.0 - fees.bet_proportion
        if fees.absolute > 0:
            amount_before_fee += fees.absolute

        return amount_before_fee

    def _calculate_revenue_from_sell_token(
        self,
        *,
        pool_size: float,
        probability: float,
        tokens_to_sell: float,
        fees: MarketFees | None = None,
        alpha: float | None = None,
    ) -> float | None:
        """Estimate collateral received when dumping `tokens_to_sell` into the
        pool. Mirrors `_calculate_cost_to_buy_token` but for the sell side.

        We model Uniswap-V3 slippage by assuming the *price impact* is the same
        functional form, i.e. the marginal price decreases with
        `(tokens_to_sell / pool_size)` raised to `alpha`.
        """

        alpha = alpha if alpha is not None else self.slippage_alpha
        fees = fees or MarketFees.get_zero_fees()

        if tokens_to_sell <= 0:
            return None

        # Approximate average price obtained when selling `tokens_to_sell`.
        # We integrate marginal price from 0..tokens_to_sell under the same
        # functional form as the buy curve.
        # Closed-form: revenue = prob * pool * (1 - (1 + x / pool) ^ -alpha )
        # where x = tokens_to_sell.
        revenue_after_fee = probability * pool_size * (1 - (1 + tokens_to_sell / pool_size) ** -alpha)

        # Add fees (the pool *subtracts* them, so we need to subtract here).
        revenue_before_fee = revenue_after_fee * (1 - fees.bet_proportion)
        if fees.absolute:
            revenue_before_fee -= fees.absolute

        return max(revenue_before_fee, 0.0)


    def max_sets_to_buy(self, market: AgentMarket) -> int:
        """
        Simulate buying complete sets one-by-one, updating an in-memory copy of
        the pools. Stop when the marginal cost of the next set is ≥ 1 USD
        (i.e. no more arbitrage).
        """
        if not market.outcome_token_pool:
            return 0

        # Local copy of pool sizes (floats for quick maths)
        pools: dict[OutcomeStr, float] = {
            o: t.value for o, t in market.outcome_token_pool.items()
        }
        probs = {o: float(p) for o, p in market.probabilities.items()}

        quantity      = 0
        MAX_QUANTITY  = 500  # safety guard

        while quantity < MAX_QUANTITY:
            marginal_cost = 0.0

            # Estimate cost of ONE additional complete set at current state
            for outcome in market.outcomes:
                cost = self._calculate_cost_to_buy_token(
                    pool_size     = pools[outcome],
                    probability   = probs[outcome],
                    tokens_to_buy = 1,
                    fees          = market.fees,
                )
                if cost is None:
                    marginal_cost = None
                    break
                marginal_cost += cost

            # Pricing failed or insufficient liquidity
            if marginal_cost is None:
                break

            # No more profit if cost ≥ payoff
            if marginal_cost >= 1.0 - self.epsilon:
                break

            # “Buy” the set – update simulated pools
            for outcome in market.outcomes:
                pools[outcome] -= 1

            quantity += 1

        logger.info(f"Optimal complete sets to buy: {quantity}")
        return quantity
    
    def _buy_complete_sets(self, market: AgentMarket, quantity: int) -> list[PlacedTrade]:
        logger.info(f"Buying {quantity} complete sets for market {market.question}")
        
        placed_trades: list[PlacedTrade] = []
        try:
            for outcome in market.outcomes:
                probability = market.probabilities[outcome]
                amount_for_outcome = USD(quantity * probability)
                
                expected_tokens = market.get_buy_token_amount(amount_for_outcome, outcome)
                expected_profit = USD(float(expected_tokens.value)) - amount_for_outcome

                logger.info(f"Buying {amount_for_outcome} worth of '{outcome}' tokens (profit: {expected_profit:.6f})")
                trade_id = market.buy_tokens(outcome=outcome, amount=amount_for_outcome)

                placed_trade = PlacedTrade(
                    trade_type=TradeType.BUY,
                    outcome=outcome,
                    amount=amount_for_outcome,
                    id=trade_id
                )
                placed_trades.append(placed_trade)
                logger.info(f"Trade executed with ID: {trade_id}")

        except Exception as e:
            logger.error(f"Failed to buy complete sets: {e}")
            raise ValueError(f"Failed to execute complete set purchase: {e}")
        
        return placed_trades

    def _get_buy_price(
        self, 
        market: SeerAgentMarket, 
        outcome: OutcomeStr, 
        amount_to_spend: Wei,
        from_address: ChecksumAddress,
    ) -> Wei | None:
        try:
            outcome_token_address = market.get_wrapped_token_for_outcome(outcome)
            collateral_token_address = market.collateral_token_contract_address_checksummed
            
            params = ExactInputSingleParams(
                token_in=collateral_token_address,
                token_out=outcome_token_address,
                recipient=from_address,
                amount_in=amount_to_spend,
                amount_out_minimum=Wei(0),  # We just want to simulate, not enforce minimum
            )
            
            return self.swapr_router.calc_exact_input_single(
                params=params,
                from_address=from_address,
                api_keys=self.api_keys
            )
            
        except Exception as e:
            logger.warning(f"Failed to get accurate buy price for {outcome}: {e}")
            return None

    def _arbitrage_overestimated(self, market: AgentMarket) -> list[PlacedTrade]:
        logger.info("Executing mint-and-sell arbitrage for over-estimated probabilities")
        
        placed_trades: list[PlacedTrade] = []
        
        try:
            optimal_sets = self.max_sets_to_mint(market)
            if optimal_sets <= 0:
                logger.info("No profitable minting opportunity found")
                return placed_trades
            
            available_balance = market.get_trade_balance(self.api_keys)
            required_balance = market.get_in_usd(CollateralToken(optimal_sets))
            
            if available_balance < required_balance:
                logger.warning(f"Insufficient balance: need {required_balance}, have {available_balance}")
                # Scale down to available balance
                affordable_sets = int(available_balance.value)
                if affordable_sets <= 0:
                    return placed_trades
                mint_amount = CollateralToken(affordable_sets)
                logger.info(f"Scaling down to affordable amount: {affordable_sets} sets")
            
            logger.info(f"Minting {mint_amount.value} complete sets for arbitrage")
            mint_receipt = market.mint_full_set_of_outcome_tokens(
                amount=mint_amount,
                api_keys=self.api_keys
            )
            tokens_to_sell = OutcomeToken(mint_amount.value)
            placed_trades.extend(self._sell_complete_sets(market, optimal_sets))
            
            return placed_trades
                
        except Exception as e:
            logger.warning(f"Over-estimated arbitrage failed: {e}")
            raise ValueError(f"Failed to execute over-estimated arbitrage: {e}")
        
        return placed_trades

    def max_sets_to_mint(self, market: AgentMarket) -> int:
        """Find how many complete sets we can *mint & immediately sell* while
        each additional set still yields > 1 USD after slippage and fees."""

        if not market.outcome_token_pool:
            return 0

        pools: dict[OutcomeStr, float] = {
            o: t.value for o, t in market.outcome_token_pool.items()
        }
        probs = {o: float(p) for o, p in market.probabilities.items()}

        quantity = 0
        MAX_QUANTITY = 500

        while quantity < MAX_QUANTITY:
            marginal_revenue = 0.0

            # Revenue from dumping one more complete set into the pool.
            for outcome in market.outcomes:
                revenue = self._calculate_revenue_from_sell_token(
                    pool_size      = pools[outcome],
                    probability    = probs[outcome],
                    tokens_to_sell = 1,
                    fees           = market.fees,
                )
                if revenue is None:
                    marginal_revenue = None
                    break
                marginal_revenue += revenue

            if marginal_revenue is None:
                break

            # No profit once selling a set brings ≤ 1 USD.
            if marginal_revenue <= 1.0 + self.epsilon:
                break

            # “Sell” the set – enlarge simulated pools.
            for outcome in market.outcomes:
                pools[outcome] += 1

            quantity += 1

        logger.info(f"Optimal complete sets to mint: {quantity}")
        return quantity

    def _sell_complete_sets(self, market: AgentMarket, quantity: int) -> list[PlacedTrade]:
        logger.info(f"Selling {quantity} complete sets to realize arbitrage profit")
        
        placed_trades: list[PlacedTrade] = []
        
        try:
            user_id = market.get_user_id(self.api_keys)
            current_position = market.get_position(user_id)

            for outcome in market.outcomes:
                tokens_to_sell = OutcomeToken(quantity * market.probabilities[outcome])
                expected_sell_value = market.get_sell_value_of_outcome_token(outcome, tokens_to_sell)
                
                amount_usd = market.get_in_usd(expected_sell_value)
                logger.info(f"Selling {tokens_to_sell} '{outcome}' tokens for {amount_usd}")
                trade_id = market.sell_tokens(outcome=outcome, amount=tokens_to_sell)
                
                placed_trade = PlacedTrade(
                    trade_type=TradeType.SELL,
                    outcome=outcome,
                    amount=amount_usd,
                    id=trade_id
                )
                placed_trades.append(placed_trade)


        except Exception as e:
            logger.warning(f"Failed to sell complete sets immediately: {e}")
            raise ValueError(f"Failed to realize arbitrage profit: {e}")
        
        return placed_trades

    def _get_sell_price(
        self, 
        market: SeerAgentMarket, 
        outcome: OutcomeStr, 
        tokens_to_sell: Wei,
        from_address: ChecksumAddress,
    ) -> Wei | None:
        try:
            outcome_token_address = market.get_wrapped_token_for_outcome(outcome)
            collateral_token_address = market.collateral_token_contract_address_checksummed
            
            params = ExactInputSingleParams(
                token_in=outcome_token_address,
                token_out=collateral_token_address,
                recipient=from_address,
                amount_in=tokens_to_sell,
                amount_out_minimum=Wei(0),  
            )
            

            return self.swapr_router.calc_exact_input_single(
                params=params,
                from_address=from_address,
                api_keys=self.api_keys
            )
            
        except Exception as e:
            logger.warning(f"Failed to get accurate sell price for {outcome}: {e}")
            return None
