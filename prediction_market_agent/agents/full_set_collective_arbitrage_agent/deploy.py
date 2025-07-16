import typing as t
import hashlib
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.agent_market import AgentMarket, ProcessedTradedMarket
from prediction_market_agent_tooling.markets.markets import OutcomeStr
from prediction_market_agent_tooling.markets.probabilistic_answer import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.probability import Probability
from prediction_market_agent_tooling.markets.data_models import CategoricalProbabilisticAnswer, Trade, PlacedTrade, TradeType, USD
from prediction_market_agent_tooling.tools.tokens.slippage import get_slippage_tolerance_per_token
from prediction_market_agent_tooling.gtypes import CollateralToken, OutcomeToken
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.config import APIKeys


class DeployableFullSetCollectiveArbitrageAgent(DeployableTraderAgent):
    def __init__(self):
        super().__init__()
        self.epsilon = 0.003  # 0.3% error margin for probabilities as rounding error

    def run(self, market_type: MarketType) -> None:
        if market_type != MarketType.OMEN:
            raise RuntimeError(
                "Can arbitrage only on Omen since related markets embeddings available only for Omen markets."
            )
        super().run(market_type=market_type)

    def get_markets(
        self,
        market_type: MarketType,
    ) -> t.Sequence[AgentMarket]:
        cls = market_type.market_class

        # Fetch all markets to choose from
        available_markets = cls.get_markets(
            limit=self.n_markets_to_fetch,
            sort_by=self.get_markets_sort_by,
            filter_by=self.get_markets_filter_by,
            created_after=self.trade_on_markets_created_after,
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
        over_estimated_prob = prob_summed > 1 + self.epsilon
        under_estimated_prob = prob_summed < 1 - self.epsilon

        if over_estimated_prob or under_estimated_prob:
            logger.info(f"Market '{market.question}' has probabilities that don't sum to 1. Sum: {prob_summed}")

            try:
                placed_trades: list[PlacedTrade] = []
                arbitrage_type = ""
                
                if under_estimated_prob:
                    logger.info("Under-estimated probabilities detected - buy low, sell high arbitrage")
                    arbitrage_type = "under-estimated"
                    max_sets_to_arbitrage = self.max_sets_to_buy(market)
                    if max_sets_to_arbitrage > 0:
                        # Buy underpriced outcome tokens
                        buy_trades = self._buy_complete_sets(market, max_sets_to_arbitrage)
                        if buy_trades:
                            placed_trades.extend(buy_trades)
                            
                            # Immediately sell those tokens to realize profit
                            sell_trades = self._sell_complete_sets(market, max_sets_to_arbitrage)
                            placed_trades.extend(sell_trades)

                if over_estimated_prob:
                    logger.info("Over-estimated probabilities detected - mint and sell arbitrage")
                    arbitrage_type = "over-estimated"
                    # For over-estimated: mint complete sets (cost 1.0) then sell (get > 1.0)
                    sell_trades = self._arbitrage_overestimated(market)
                    placed_trades.extend(sell_trades)
                
                # Create arbitrage answer
                answer = CategoricalProbabilisticAnswer(
                    probabilities=market.probabilities,
                    confidence=1.0,
                    reasoning=f"Arbitrage opportunity detected: {arbitrage_type} probabilities (sum: {prob_summed:.4f}). Executed {len(placed_trades)} trades for immediate profit realization."
                )
                
                return ProcessedTradedMarket(
                    answer=answer,
                    trades=placed_trades
                )
                
            except Exception as e:
                logger.error(f"Market '{market.question}' - skipping due to error {e}")
                return None
        
        # No arbitrage opportunity
        return None
                
    def max_sets_to_buy(self, market: AgentMarket) -> int:
        """
        Calculate the maximum number of complete sets to buy before arbitrage opportunity disappears.
        
        For under-estimated probabilities (sum < 1), we can buy complete sets for less than 1 unit
        but they always pay out 1 unit, creating risk-free profit.
        
        This function accounts for:
        1. Market impact (slippage) as we buy more tokens
        2. Fees
        3. Limited liquidity in outcome token pools
        
        Note: This assumes outcome_token_pool has been validated in process_market
        """
        quantity = 0
        
        while True:
            # Calculate marginal cost of buying the (quantity+1)-th complete set
            marginal_cost = 0
            
            for outcome in market.outcomes:
                prob = market.probabilities[outcome]
                pool_size = market.outcome_token_pool[outcome].value
                
                # Check if there's enough liquidity - if not, raise error - it does not make sense to mint
                # because price will be 1 and we will loose edge that we have for this arbitrage opportunity
                if pool_size <= quantity + 1:
                    raise ValueError(f"Insufficient liquidity for outcome '{outcome}' (pool size: {pool_size}, needed: {quantity + 1})")
                
                slippage_tolerance = get_slippage_tolerance_per_token(
                    market.collateral_token_contract_address_checksummed, 
                    outcome
                )
                
                # Calculate price impact: buying (quantity+1) tokens from a pool
                # Using constant product market maker formula approximation
                price_with_slippage = prob * (1 + slippage_tolerance)
                price_impact = price_with_slippage * (pool_size / (pool_size - (quantity + 1)))
                
                marginal_cost += price_impact
            
            # Add fees to marginal cost
            marginal_cost *= (1 + market.fees.bet_proportion)
            marginal_cost += market.fees.absolute.value if market.fees.absolute else 0
            
            # If marginal cost >= 1 - epsilon, arbitrage opportunity is gone
            if marginal_cost >= 1 - self.epsilon:
                break
                
            quantity += 1
            
            # Safety check to prevent infinite loops
            if quantity > 1000:
                logger.warning("Hit safety limit of 1000 sets")
                break
                
        return quantity
    
    def max_sets_to_mint(self, market: AgentMarket) -> int:
        """
        Calculate the optimal number of complete sets to mint for over-estimated arbitrage.
        
        For over-estimated probabilities (sum > 1), we:
        1. Mint complete sets (cost: quantity * 1.0)
        2. Sell all outcome tokens (revenue: quantity * sum(probabilities) - slippage)
        3. Profit = revenue - cost
        
        Returns optimal quantity where marginal profit = marginal cost increase
        """
        quantity = 0
        
        while True:
            # Calculate marginal revenue from selling the (quantity+1)-th complete set
            marginal_revenue = 0
            
            for outcome in market.outcomes:
                prob = market.probabilities[outcome]
                pool_size = market.outcome_token_pool[outcome].value
                
                # Check liquidity constraint - we need to be able to sell tokens
                if pool_size <= quantity + 1:
                    logger.warning(f"Insufficient sell liquidity for outcome '{outcome}' (pool size: {pool_size}, trying to sell: {quantity + 1})")
                    raise ValueError("Insufficient sell liquidity")
                
                # Calculate price impact when selling (quantity+1) tokens to the pool
                slippage_tolerance = get_slippage_tolerance_per_token(
                    market.collateral_token_contract_address_checksummed, 
                    outcome
                )
                
                # For selling, price decreases as we dump more tokens into the pool
                # Using AMM formula: selling reduces the price we get
                base_price = prob
                price_impact = base_price * (1 - slippage_tolerance) * (pool_size / (pool_size + (quantity + 1)))
                
                marginal_revenue += price_impact
            
            # Subtract selling fees from marginal revenue
            marginal_revenue *= (1 - market.fees.bet_proportion)
            marginal_revenue -= market.fees.absolute.value if market.fees.absolute else 0
            
            # Marginal cost is always 1.0 (cost to mint one complete set)
            marginal_cost = 1.0
            
            # If marginal revenue <= marginal cost + epsilon, stop
            if marginal_revenue <= marginal_cost + self.epsilon:
                break
                
            quantity += 1
            
            # Safety check
            if quantity > 1000:
                logger.warning("Hit safety limit of 1000 sets for minting")
                break
                
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
                
                slippage_cost = amount_for_outcome * get_slippage_tolerance_per_token(
                    market.collateral_token_contract_address_checksummed, 
                    outcome
                )
                
                if expected_profit > slippage_cost:
                    logger.info(f"Buying {amount_for_outcome} worth of '{outcome}' tokens (profit: {expected_profit}, costs: {total_costs})")
                    trade_id = market.buy_tokens(outcome=outcome, amount=amount_for_outcome)
                    
                    placed_trade = PlacedTrade(
                        trade_type=TradeType.BUY,
                        outcome=outcome,
                        amount=amount_for_outcome,
                        id=trade_id
                    )
                    placed_trades.append(placed_trade)
                    logger.info(f"Trade executed with ID: {trade_id}")
                else:
                    logger.info(f"Skipping outcome '{outcome}' - insufficient profit: {expected_profit} vs costs: {slippage_cost}")
                    
        except Exception as e:
            logger.error(f"Failed to buy complete sets: {e}")
            raise ValueError(f"Failed to execute complete set purchase: {e}")
        
        return placed_trades
    
    def _sell_complete_sets(self, market: AgentMarket, quantity: int) -> list[PlacedTrade]:
        logger.info(f"Selling {quantity} complete sets to realize arbitrage profit")
        
        placed_trades: list[PlacedTrade] = []
        
        try:
            api_keys = APIKeys()
            user_id = market.get_user_id(api_keys)
            current_position = market.get_position(user_id)

            for outcome in market.outcomes:
                tokens_to_sell = OutcomeToken(quantity * market.probabilities[outcome])
                expected_sell_value = market.get_sell_value_of_outcome_token(outcome, tokens_to_sell)
                amount_usd = market.get_in_usd(expected_sell_value)
                
                if amount_usd > USD(0.001):  # Meaningful amount
                    logger.info(f"Selling {tokens_to_sell} '{outcome}' tokens for {amount_usd}")
                    trade_id = market.sell_tokens(outcome=outcome, amount=tokens_to_sell)
                    
                    placed_trade = PlacedTrade(
                        trade_type=TradeType.SELL,
                        outcome=outcome,
                        amount=amount_usd,
                        id=trade_id
                    )
                    placed_trades.append(placed_trade)
                else:
                    logger.debug(f"Skipping sell of '{outcome}' - amount too small: {amount_usd}")

        except Exception as e:
            logger.warning(f"Failed to sell complete sets immediately: {e}")
            raise ValueError(f"Failed to realize arbitrage profit: {e}")
        
        return placed_trades

    def _arbitrage_overestimated(self, market: AgentMarket) -> list[PlacedTrade]:

        logger.info("Executing mint-and-sell arbitrage for over-estimated probabilities")
        
        placed_trades: list[PlacedTrade] = []
        
        try:
            optimal_sets = self.max_sets_to_mint(market)
            if optimal_sets <= 0:
                logger.info("No profitable minting opportunity found")
                return placed_trades
            
            api_keys = APIKeys()
            available_balance = market.get_trade_balance(api_keys)
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
                api_keys=api_keys
            )
            tokens_to_sell = OutcomeToken(mint_amount.value) # TODO: check if this is correct
            placed_trades.extend(self._sell_complete_sets(market, optimal_sets))
            
            return placed_trades
                
        except Exception as e:
            logger.warning(f"Over-estimated arbitrage failed: {e}")
            raise ValueError(f"Failed to execute over-estimated arbitrage: {e}")
        
        return placed_trades