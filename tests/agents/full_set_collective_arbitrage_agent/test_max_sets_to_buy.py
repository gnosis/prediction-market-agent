import pytest
import math
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal

from prediction_market_agent_tooling.gtypes import (
    OutcomeStr, OutcomeToken, Probability, USD, CollateralToken, Wei, ChecksumAddress
)
from prediction_market_agent_tooling.markets.market_fees import MarketFees
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.seer.seer import SeerAgentMarket
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent.agents.full_set_collective_arbitrage_agent.deploy import (
    DeployableFullSetCollectiveArbitrageAgent,
)


# --------------------------------------------------------------------------- #
# Realistic Uniswap V3 Market Mocks                                           #
# --------------------------------------------------------------------------- #

class RealisticUniswapV3MockMarket:
    """
    Realistic mock for SeerAgentMarket with Uniswap V3 pricing mechanics.
    Simulates actual price impact and liquidity dynamics.
    """
    
    def __init__(
        self,
        outcomes: list[OutcomeStr],
        probabilities: dict[OutcomeStr, Probability],
        liquidity_per_outcome: float = 10000.0,  # USD liquidity per outcome
        fees: MarketFees | None = None,
        collateral_token_usd_price: float = 1.02,  # Realistic price for yield-bearing tokens like sDAI
    ):
        self.outcomes = outcomes
        self.probabilities = probabilities
        self.liquidity_per_outcome = liquidity_per_outcome
        self.fees = fees or MarketFees(bet_proportion=0.02, absolute=USD(0.1))
        self.collateral_token_usd_price = collateral_token_usd_price
        
        # Simulate realistic outcome token pools based on probabilities
        self.outcome_token_pool = {}
        for outcome in outcomes:
            # Pool sizes inversely related to probability (higher prob = less tokens available)
            pool_size = liquidity_per_outcome / max(probabilities[outcome], 0.1)
            self.outcome_token_pool[outcome] = OutcomeToken(pool_size)
    
    def get_token_in_usd(self, collateral_amount: CollateralToken) -> USD:
        """Convert CollateralToken to USD using realistic exchange rate."""
        return USD(collateral_amount.value * self.collateral_token_usd_price)
    
    def get_usd_in_token(self, usd_amount: USD) -> CollateralToken:
        """Convert USD to CollateralToken using realistic exchange rate."""
        return CollateralToken(usd_amount / self.collateral_token_usd_price)
    
    def get_in_token(self, amount: USD | CollateralToken) -> CollateralToken:
        """Convert USD to CollateralToken if needed."""
        if isinstance(amount, USD):
            return self.get_usd_in_token(amount)
        return amount
    
    def get_buy_token_amount(self, bet_amount: USD | CollateralToken, outcome: OutcomeStr) -> OutcomeToken:
        """
        Simulates Uniswap V3 constant product formula with realistic price impact.
        Price impact increases with trade size relative to liquidity.
        """
        # Convert to consistent units (CollateralToken)
        if isinstance(bet_amount, USD):
            amount_in_collateral = self.get_usd_in_token(bet_amount)
        else:
            amount_in_collateral = bet_amount
            
        amount_value = amount_in_collateral.value
            
        # Apply fees first
        amount_after_fees = amount_value * (1 - self.fees.bet_proportion)
        if self.fees.absolute:
            absolute_fee_in_collateral = self.get_usd_in_token(self.fees.absolute)
            amount_after_fees -= absolute_fee_in_collateral.value
            
        # Get current pool size for outcome
        current_pool = self.outcome_token_pool[outcome].value
        probability = self.probabilities[outcome]
        
        # Uniswap V3 pricing with price impact
        # More realistic formula: tokens_out = pool * (1 - (pool / (pool + amount_in))^α)
        # where α controls price impact (closer to 1 = more impact)
        alpha = 0.8  # Price impact factor
        
        # Calculate tokens received with price impact
        pool_ratio = current_pool / (current_pool + amount_after_fees / probability)
        tokens_out = current_pool * (1 - pool_ratio ** alpha)
        
        # Ensure we don't exceed available liquidity
        tokens_out = min(tokens_out, current_pool * 0.95)
        
        return OutcomeToken(max(0, tokens_out))
    
    def get_sell_value_of_outcome_token(
        self, outcome: OutcomeStr, amount: OutcomeToken
    ) -> CollateralToken:
        """
        Simulates selling outcome tokens back to Uniswap V3 pool.
        """
        if amount.value == 0:
            return CollateralToken(0)
            
        current_pool = self.outcome_token_pool[outcome].value
        probability = self.probabilities[outcome]
        
        # Reverse of buy calculation - selling reduces token supply
        # collateral_out = amount_tokens * probability * (1 - slippage)
        slippage_factor = min(amount.value / current_pool * 0.1, 0.05)  # Max 5% slippage
        base_value = amount.value * probability * (1 - slippage_factor)
        
        # Apply fees
        after_fees = base_value * (1 - self.fees.bet_proportion)
        if self.fees.absolute:
            absolute_fee_in_collateral = self.get_usd_in_token(self.fees.absolute)
            after_fees -= absolute_fee_in_collateral.value
            
        return CollateralToken(max(0, after_fees))


@pytest.fixture
def realistic_underestimated_market() -> AgentMarket:
    """
    Market with probabilities summing to < 1 (underestimated scenario).
    Yes: 0.40, No: 0.45 = 0.85 total (15% arbitrage opportunity)
    """
    probabilities = {
        OutcomeStr("Yes"): Probability(0.40),
        OutcomeStr("No"): Probability(0.45),
    }
    
    mock_market = Mock(spec=SeerAgentMarket)
    realistic_market = RealisticUniswapV3MockMarket(
        outcomes=[OutcomeStr("Yes"), OutcomeStr("No")],
        probabilities=probabilities,
        liquidity_per_outcome=50000.0,  # $50k liquidity per outcome
        fees=MarketFees(bet_proportion=0.015, absolute=USD(0.05)),  # 1.5% + $0.05
        collateral_token_usd_price=1.025,  # sDAI with 2.5% yield
    )
    
    # Setup mock market properties
    mock_market.question = "Will event happen?"
    mock_market.outcomes = realistic_market.outcomes
    mock_market.probabilities = probabilities
    mock_market.outcome_token_pool = realistic_market.outcome_token_pool
    mock_market.fees = realistic_market.fees
    
    # Mock the pricing methods with realistic implementations
    mock_market.get_buy_token_amount = realistic_market.get_buy_token_amount
    mock_market.get_sell_value_of_outcome_token = realistic_market.get_sell_value_of_outcome_token
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    
    # Mock other required methods
    mock_market.get_liquidity = Mock(return_value=CollateralToken(97561))  # ~100k USD / 1.025
    
    return mock_market


@pytest.fixture
def realistic_overestimated_market() -> AgentMarket:
    """
    Market with probabilities summing to > 1 (overestimated scenario).
    Yes: 0.65, No: 0.55 = 1.20 total (20% arbitrage opportunity)
    """
    probabilities = {
        OutcomeStr("Yes"): Probability(0.65),
        OutcomeStr("No"): Probability(0.55),
    }
    
    mock_market = Mock(spec=SeerAgentMarket)
    realistic_market = RealisticUniswapV3MockMarket(
        outcomes=[OutcomeStr("Yes"), OutcomeStr("No")],
        probabilities=probabilities,
        liquidity_per_outcome=30000.0,  # $30k liquidity per outcome
        fees=MarketFees(bet_proportion=0.02, absolute=USD(0.1)),  # 2% + $0.10
        collateral_token_usd_price=1.018,  # Different sDAI rate
    )
    
    # Setup mock market properties
    mock_market.question = "Will another event happen?"
    mock_market.outcomes = realistic_market.outcomes
    mock_market.probabilities = probabilities
    mock_market.outcome_token_pool = realistic_market.outcome_token_pool
    mock_market.fees = realistic_market.fees
    
    # Mock the pricing methods with realistic implementations
    mock_market.get_buy_token_amount = realistic_market.get_buy_token_amount
    mock_market.get_sell_value_of_outcome_token = realistic_market.get_sell_value_of_outcome_token
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    
    # Mock other required methods
    mock_market.get_liquidity = Mock(return_value=CollateralToken(58928))  # ~60k USD / 1.018
    
    return mock_market


@pytest.fixture
def agent() -> DeployableFullSetCollectiveArbitrageAgent:
    """Agent fixture with realistic mocks."""
    agent = DeployableFullSetCollectiveArbitrageAgent()
    agent.epsilon = 0.015  # 1.5% error margin
    
    # Mock the SwaprRouter for price calculations
    agent.swapr_router = Mock()
    agent.api_keys = Mock(spec=APIKeys)
    agent.api_keys.bet_from_address = ChecksumAddress("0x1234567890123456789012345678901234567890")
    
    return agent


# --------------------------------------------------------------------------- #
# Tests for Underestimated Probabilities (sum < 1) - Buy Low, Sell High      #
# --------------------------------------------------------------------------- #

def test_max_sets_to_buy_underestimated_simple(agent, realistic_underestimated_market):
    """
    Test max_sets_to_buy for underestimated probabilities.
    Should find optimal quantity where marginal cost approaches 1.0.
    """
    # Mock the precise pricing method
    def mock_get_buy_price(market, outcome, amount_wei, from_address):
        # Simulate realistic Uniswap V3 price with slippage
        amount_collateral = CollateralToken(amount_wei.value / 10**18)
        tokens_received = market.get_buy_token_amount(amount_collateral, outcome)
        return Wei(int(tokens_received.value * 10**18))
    
    agent._get_buy_price = mock_get_buy_price
    
    # Test the calculation
    max_sets = agent.max_sets_to_buy(realistic_underestimated_market)
    
    # With underestimated probabilities (0.85 total), we should find profitable opportunities
    assert max_sets > 0, "Should find profitable arbitrage opportunity"
    assert max_sets <= 50, "Should be reasonable quantity given liquidity constraints"
    
    # Verify the marginal cost calculation makes sense
    # For underestimated markets, buying complete sets should cost less than 1.0
    total_prob = sum(realistic_underestimated_market.probabilities.values())
    expected_opportunity = 1.0 - total_prob  # 0.15 in this case
    
    assert expected_opportunity > agent.epsilon, "Should detect significant arbitrage opportunity"


def test_max_sets_to_buy_with_high_liquidity(agent):
    """Test behavior with very high liquidity markets."""
    # Create market with high liquidity to minimize price impact
    probabilities = {
        OutcomeStr("Yes"): Probability(0.35),
        OutcomeStr("No"): Probability(0.45),  # 0.80 total
    }
    
    mock_market = Mock(spec=SeerAgentMarket)
    realistic_market = RealisticUniswapV3MockMarket(
        outcomes=[OutcomeStr("Yes"), OutcomeStr("No")],
        probabilities=probabilities,
        liquidity_per_outcome=500000.0,  # $500k liquidity - very high
        fees=MarketFees(bet_proportion=0.01, absolute=USD(0.02)),  # Low fees
        collateral_token_usd_price=0.999,  # xDAI slightly below parity
    )
    
    mock_market.outcomes = realistic_market.outcomes
    mock_market.probabilities = probabilities
    mock_market.outcome_token_pool = realistic_market.outcome_token_pool
    mock_market.fees = realistic_market.fees
    mock_market.get_buy_token_amount = realistic_market.get_buy_token_amount
    mock_market.get_sell_value_of_outcome_token = realistic_market.get_sell_value_of_outcome_token
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    mock_market.get_liquidity = Mock(return_value=CollateralToken(1001001))  # ~1M USD / 0.999
    
    def mock_get_buy_price(market, outcome, amount_wei, from_address):
        amount_collateral = CollateralToken(amount_wei.value / 10**18)
        tokens_received = market.get_buy_token_amount(amount_collateral, outcome)
        return Wei(int(tokens_received.value * 10**18))
    
    agent._get_buy_price = mock_get_buy_price
    
    max_sets = agent.max_sets_to_buy(mock_market)
    
    # High liquidity should allow larger arbitrage quantities
    assert max_sets > 10, "High liquidity should enable larger arbitrage trades"


def test_max_sets_to_buy_no_arbitrage_opportunity(agent):
    """Test that no arbitrage is detected when probabilities sum to ~1."""
    # Balanced probabilities
    probabilities = {
        OutcomeStr("Yes"): Probability(0.51),
        OutcomeStr("No"): Probability(0.49),  # 1.00 total
    }
    
    mock_market = Mock(spec=SeerAgentMarket)
    realistic_market = RealisticUniswapV3MockMarket(
        outcomes=[OutcomeStr("Yes"), OutcomeStr("No")],
        probabilities=probabilities,
        liquidity_per_outcome=50000.0,
        collateral_token_usd_price=1.005,  # Small premium on collateral
    )
    
    mock_market.outcomes = realistic_market.outcomes
    mock_market.probabilities = probabilities
    mock_market.outcome_token_pool = realistic_market.outcome_token_pool
    mock_market.fees = realistic_market.fees
    mock_market.get_buy_token_amount = realistic_market.get_buy_token_amount
    mock_market.get_sell_value_of_outcome_token = realistic_market.get_sell_value_of_outcome_token
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    mock_market.get_liquidity = Mock(return_value=CollateralToken(99502))  # ~100k USD / 1.005
    
    def mock_get_buy_price(market, outcome, amount_wei, from_address):
        amount_collateral = CollateralToken(amount_wei.value / 10**18)
        tokens_received = market.get_buy_token_amount(amount_collateral, outcome)
        return Wei(int(tokens_received.value * 10**18))
    
    agent._get_buy_price = mock_get_buy_price
    
    max_sets = agent.max_sets_to_buy(mock_market)
    
    # Should find no profitable arbitrage when probabilities sum to 1
    assert max_sets == 0, "Should find no arbitrage opportunity when probabilities are balanced"


# --------------------------------------------------------------------------- #
# Tests for Overestimated Probabilities (sum > 1) - Mint and Sell           #
# --------------------------------------------------------------------------- #

def test_max_sets_to_mint_overestimated_simple(agent, realistic_overestimated_market):
    """
    Test max_sets_to_mint for overestimated probabilities.
    Should find optimal quantity where marginal sell value approaches 1.0.
    """
    # Mock the precise pricing method for selling
    def mock_get_sell_price(market, outcome, tokens_to_sell_wei, from_address):
        # Convert wei to tokens and get sell value
        tokens_to_sell = OutcomeToken(tokens_to_sell_wei.value / 10**18)
        collateral_received = market.get_sell_value_of_outcome_token(outcome, tokens_to_sell)
        return Wei(int(collateral_received.value * 10**18))
    
    agent._get_sell_price = mock_get_sell_price
    
    # Test the calculation
    max_sets = agent.max_sets_to_mint(realistic_overestimated_market)
    
    # With overestimated probabilities (1.20 total), we should find profitable opportunities
    assert max_sets > 0, "Should find profitable mint-and-sell arbitrage opportunity"
    assert max_sets <= 100, "Should be reasonable quantity given liquidity constraints"
    
    # Verify the arbitrage opportunity exists
    total_prob = sum(realistic_overestimated_market.probabilities.values())
    expected_opportunity = total_prob - 1.0  # 0.20 in this case
    
    assert expected_opportunity > agent.epsilon, "Should detect significant arbitrage opportunity"


def test_max_sets_to_mint_with_price_impact(agent):
    """Test mint calculations with realistic price impact from large trades."""
    # Create market with lower liquidity to test price impact
    probabilities = {
        OutcomeStr("Yes"): Probability(0.70),
        OutcomeStr("No"): Probability(0.60),  # 1.30 total - large opportunity
    }
    
    mock_market = Mock(spec=SeerAgentMarket)
    realistic_market = RealisticUniswapV3MockMarket(
        outcomes=[OutcomeStr("Yes"), OutcomeStr("No")],
        probabilities=probabilities,
        liquidity_per_outcome=20000.0,  # Lower liquidity
        fees=MarketFees(bet_proportion=0.025, absolute=USD(0.15)),  # Higher fees
        collateral_token_usd_price=1.035,  # Higher yield sDAI
    )
    
    mock_market.outcomes = realistic_market.outcomes
    mock_market.probabilities = probabilities
    mock_market.outcome_token_pool = realistic_market.outcome_token_pool
    mock_market.fees = realistic_market.fees
    mock_market.get_buy_token_amount = realistic_market.get_buy_token_amount
    mock_market.get_sell_value_of_outcome_token = realistic_market.get_sell_value_of_outcome_token
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    mock_market.get_liquidity = Mock(return_value=CollateralToken(38647))  # ~40k USD / 1.035
    
    def mock_get_sell_price(market, outcome, tokens_to_sell_wei, from_address):
        tokens_to_sell = OutcomeToken(tokens_to_sell_wei.value / 10**18)
        collateral_received = market.get_sell_value_of_outcome_token(outcome, tokens_to_sell)
        return Wei(int(collateral_received.value * 10**18))
    
    agent._get_sell_price = mock_get_sell_price
    
    max_sets = agent.max_sets_to_mint(mock_market)
    
    # Price impact should limit the arbitrage size despite large opportunity
    assert max_sets > 0, "Should still find arbitrage despite price impact"
    assert max_sets < 50, "Price impact should limit arbitrage size"


def test_estimate_arbitrage_scale(agent, realistic_underestimated_market):
    """Test the arbitrage scale estimation method."""
    scale = agent.estimate_arbitrage_scale(realistic_underestimated_market)
    
    assert scale > 0, "Should estimate positive arbitrage scale"
    assert isinstance(scale, (int, float)), "Should return numeric scale"
    
    # Scale should be related to liquidity and arbitrage gap
    arbitrage_gap = abs(1.0 - sum(realistic_underestimated_market.probabilities.values()))
    assert arbitrage_gap > 0.1, "Test market should have significant arbitrage gap"


def test_collateral_token_pricing_realism(realistic_underestimated_market):
    """Test that collateral token pricing is realistic and consistent."""
    # Test USD to token conversion
    usd_amount = USD(100)
    token_amount = realistic_underestimated_market.get_usd_in_token(usd_amount)
    
    # Should get fewer tokens when collateral trades at premium
    assert token_amount.value < 100, "Should get fewer tokens when collateral is above $1"
    
    # Test round-trip conversion
    converted_back = realistic_underestimated_market.get_token_in_usd(token_amount)
    assert abs(converted_back.value - usd_amount.value) < 0.01, "Round-trip conversion should be accurate"
    
    # Test realistic range
    exchange_rate = converted_back.value / token_amount.value
    assert 0.98 <= exchange_rate <= 1.10, f"Exchange rate {exchange_rate} should be realistic for stablecoins/yield tokens"


# --------------------------------------------------------------------------- #
# Edge Cases and Error Handling                                              #
# --------------------------------------------------------------------------- #

def test_max_sets_to_buy_zero_liquidity(agent):
    """Test behavior with zero liquidity pools."""
    probabilities = {
        OutcomeStr("Yes"): Probability(0.40),
        OutcomeStr("No"): Probability(0.50),
    }
    
    mock_market = Mock(spec=SeerAgentMarket)
    realistic_market = RealisticUniswapV3MockMarket(
        outcomes=[OutcomeStr("Yes"), OutcomeStr("No")],
        probabilities=probabilities,
        liquidity_per_outcome=0,  # Zero liquidity
        collateral_token_usd_price=1.01,
    )
    
    mock_market.outcomes = realistic_market.outcomes
    mock_market.probabilities = probabilities
    mock_market.outcome_token_pool = {
        OutcomeStr("Yes"): OutcomeToken(0),  # Zero liquidity
        OutcomeStr("No"): OutcomeToken(0),
    }
    mock_market.fees = realistic_market.fees
    
    # Mock pricing methods to return zero/none
    mock_market.get_buy_token_amount = Mock(return_value=OutcomeToken(0))
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    mock_market.get_liquidity = Mock(return_value=CollateralToken(0))
    
    agent._get_buy_price = Mock(return_value=Wei(0))
    
    max_sets = agent.max_sets_to_buy(mock_market)
    
    # Should handle zero liquidity gracefully
    assert max_sets == 0, "Should return 0 for markets with no liquidity"


def test_max_sets_calculation_with_extreme_probabilities(agent):
    """Test with extreme probability imbalances."""
    # Very extreme underestimation
    probabilities = {
        OutcomeStr("Yes"): Probability(0.05),  
        OutcomeStr("No"): Probability(0.10),   # 0.15 total - extreme underestimation
    }
    
    mock_market = Mock(spec=SeerAgentMarket)
    realistic_market = RealisticUniswapV3MockMarket(
        outcomes=[OutcomeStr("Yes"), OutcomeStr("No")],
        probabilities=probabilities,
        liquidity_per_outcome=100000.0,  # High liquidity to handle extreme case
        collateral_token_usd_price=0.995,  # xDAI slightly discounted
    )
    
    mock_market.outcomes = realistic_market.outcomes
    mock_market.probabilities = probabilities
    mock_market.outcome_token_pool = realistic_market.outcome_token_pool
    mock_market.fees = realistic_market.fees
    mock_market.get_buy_token_amount = realistic_market.get_buy_token_amount
    mock_market.get_sell_value_of_outcome_token = realistic_market.get_sell_value_of_outcome_token
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    mock_market.get_liquidity = Mock(return_value=CollateralToken(201005))  # ~200k USD / 0.995
    
    def mock_get_buy_price(market, outcome, amount_wei, from_address):
        amount_collateral = CollateralToken(amount_wei.value / 10**18)
        tokens_received = market.get_buy_token_amount(amount_collateral, outcome)
        return Wei(int(tokens_received.value * 10**18))
    
    agent._get_buy_price = mock_get_buy_price
    
    max_sets = agent.max_sets_to_buy(mock_market)
    
    # Even extreme arbitrage should be limited by practical constraints
    assert max_sets >= 0, "Should handle extreme cases without errors"
    assert max_sets <= 200, "Should still respect practical limits"
