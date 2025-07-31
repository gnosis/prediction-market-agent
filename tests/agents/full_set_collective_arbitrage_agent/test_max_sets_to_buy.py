import math
import typing as t
from unittest.mock import Mock

import pytest
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.gtypes import (
    USD,
    ChecksumAddress,
    CollateralToken,
    HexAddress,
    HexBytes,
    HexStr,
    OutcomeStr,
    OutcomeToken,
    Probability,
    Wei,
)
from prediction_market_agent_tooling.markets.market_fees import MarketFees
from prediction_market_agent_tooling.markets.seer.seer import SeerAgentMarket

from prediction_market_agent.agents.full_set_collective_arbitrage_agent.deploy import (
    CalculationType,
    DeployableFullSetCollectiveArbitrageAgent,
)

# --------------------------------------------------------------------------- #
# Realistic Uniswap V3 Market Mocks                                           #
# --------------------------------------------------------------------------- #


class RealisticUniswapV3MockMarket(SeerAgentMarket):
    """
    Realistic mock for SeerAgentMarket with Uniswap V3 pricing mechanics.
    Simulates actual price impact and liquidity dynamics.
    """

    # These are helper attributes, NOT Pydantic model fields
    # Using ClassVar tells Pydantic to ignore them during validation
    liquidity_per_outcome: t.ClassVar[float]
    collateral_token_usd_price: t.ClassVar[float]

    def __init__(
        self,
        outcomes: list[OutcomeStr],
        probabilities: dict[OutcomeStr, Probability],
        liquidity_per_outcome: float = 10000.0,  # USD liquidity per outcome
        fees: MarketFees | None = None,
        collateral_token_usd_price: float = 1.02,  # Realistic price for yield-bearing tokens like sDAI
    ):
        # 1. Initialise the underlying Pydantic model first
        super().__init__(
            id="mock_market_id",
            question="Mock market for testing",
            description="Mock market for testing arbitrage calculations",
            outcomes=outcomes,
            probabilities=probabilities,
            outcome_token_pool=None,  # Will be set below
            resolution=None,
            created_time=None,
            close_time=None,
            url="https://mock.market",
            volume=None,
            fees=fees or MarketFees(bet_proportion=0.02, absolute=0.01),
            wrapped_tokens=[],
            creator=HexAddress(HexStr("0x0000000000000000000000000000000000000000")),
            collateral_token_contract_address_checksummed=ChecksumAddress(
                HexAddress(HexStr("0x0000000000000000000000000000000000000000"))
            ),
            condition_id=HexBytes(b"\x00" * 32),
            outcomes_supply=0,
        )

        # 2. Attach helper attributes *after* BaseModel setup, bypassing its
        #    attribute guard. The annotations above keep the type checker happy.
        object.__setattr__(self, "liquidity_per_outcome", liquidity_per_outcome)
        object.__setattr__(
            self, "collateral_token_usd_price", collateral_token_usd_price
        )

        self.outcome_token_pool = {
            outcome: OutcomeToken(
                math.sqrt(liquidity_per_outcome / max(probabilities[outcome], 0.1))
            )
            for outcome in outcomes
        }

    def get_token_in_usd(self, collateral_amount: CollateralToken) -> USD:
        """Convert CollateralToken to USD using realistic exchange rate."""
        return USD(collateral_amount.value * self.collateral_token_usd_price)

    def get_usd_in_token(self, usd_amount: USD) -> CollateralToken:
        """Convert USD to CollateralToken using realistic exchange rate."""
        return CollateralToken(usd_amount.value / self.collateral_token_usd_price)

    def get_in_token(self, amount: USD | CollateralToken) -> CollateralToken:
        """Convert USD to CollateralToken if needed."""
        if isinstance(amount, USD):
            return self.get_usd_in_token(amount)
        return amount

    def get_buy_token_amount(
        self, bet_amount: USD | CollateralToken, outcome: OutcomeStr
    ) -> OutcomeToken:
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
            absolute_fee_in_collateral = self.get_usd_in_token(USD(self.fees.absolute))
            amount_after_fees -= absolute_fee_in_collateral.value

        # Get current pool size for outcome
        current_pool = (
            self.outcome_token_pool[outcome].value if self.outcome_token_pool else 0
        )
        probability = self.probabilities[outcome]

        # Uniswap V3 pricing with price impact
        # More realistic formula: tokens_out = pool * (1 - (pool / (pool + amount_in))^α)
        # where α controls price impact (closer to 1 = more impact)
        alpha = 0.8  # Price impact factor

        # Calculate tokens received with price impact
        pool_ratio = current_pool / (current_pool + amount_after_fees / probability)
        tokens_out = current_pool * (1 - pool_ratio**alpha)

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

        current_pool = (
            self.outcome_token_pool[outcome].value if self.outcome_token_pool else 0
        )
        probability = self.probabilities[outcome]

        # Reverse of buy calculation - selling reduces token supply
        # collateral_out = amount_tokens * probability * (1 - slippage)
        slippage_factor = min(
            amount.value / current_pool * 0.1, 0.05
        )  # Max 5% slippage
        base_value = amount.value * probability * (1 - slippage_factor)

        # Apply fees
        after_fees = base_value * (1 - self.fees.bet_proportion)
        if self.fees.absolute:
            absolute_fee_in_collateral = self.get_usd_in_token(USD(self.fees.absolute))
            after_fees -= absolute_fee_in_collateral.value

        return CollateralToken(max(0, after_fees))


@pytest.fixture
def realistic_underestimated_market() -> SeerAgentMarket:
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
        fees=MarketFees(bet_proportion=0.015, absolute=0.05),  # 1.5% + $0.05
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
    mock_market.get_sell_value_of_outcome_token = (
        realistic_market.get_sell_value_of_outcome_token
    )
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token

    # Mock other required methods
    mock_market.get_liquidity = Mock(
        return_value=CollateralToken(97561)
    )  # ~100k USD / 1.025

    return mock_market


@pytest.fixture
def realistic_overestimated_market() -> SeerAgentMarket:
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
        fees=MarketFees(bet_proportion=0.02, absolute=0.01),  # 2% + $0.01
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
    mock_market.get_sell_value_of_outcome_token = (
        realistic_market.get_sell_value_of_outcome_token
    )
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token

    # Mock other required methods
    mock_market.get_liquidity = Mock(
        return_value=CollateralToken(58928)
    )  # ~60k USD / 1.018

    return mock_market


@pytest.fixture
def agent() -> DeployableFullSetCollectiveArbitrageAgent:
    """Agent fixture with realistic mocks."""
    agent = DeployableFullSetCollectiveArbitrageAgent()
    agent.epsilon = 0.015  # 1.5% error margin

    # Mock the SwaprRouter for price calculations
    agent.swapr_router = Mock()
    agent.api_keys = Mock(spec=APIKeys)
    agent.api_keys.bet_from_address = ChecksumAddress(
        HexAddress(HexStr("0x1234567890123456789012345678901234567890"))
    )
    return agent


# --------------------------------------------------------------------------- #
# Tests for Underestimated Probabilities (sum < 1) - Buy Low, Sell High      #
# --------------------------------------------------------------------------- #


def test_max_sets_to_buy_underestimated_pivots_at_one(
    agent: DeployableFullSetCollectiveArbitrageAgent,
    realistic_underestimated_market: RealisticUniswapV3MockMarket,
) -> None:
    """
    Test that max_sets_to_buy finds the unique N where the marginal cost
    of buying one more complete set crosses 1.0.
    """
    N = agent._max_sets(realistic_underestimated_market, CalculationType.COST)
    assert N > 0, "Should detect an arbitrage opportunity"
    assert N <= 50, "Quantity should be bounded by liquidity"

    pools = (
        {
            o: t.value
            for o, t in realistic_underestimated_market.outcome_token_pool.items()
        }
        if realistic_underestimated_market.outcome_token_pool
        else {o: 0 for o in realistic_underestimated_market.outcomes}
    )
    probs = {
        o: float(p) for o, p in realistic_underestimated_market.probabilities.items()
    }
    # k_i = p_i * x_i^2
    k = {o: probs[o] * pools[o] ** 2 for o in pools}

    # 4) Compute the marginal cost at N, N - 1, and N + 1
    mc_at_N = agent._get_marginal_quote(
        N,
        k,
        pools,
        realistic_underestimated_market.fees,
        realistic_underestimated_market,
        CalculationType.COST,
    )
    mc_below = (
        agent._get_marginal_quote(
            N - 1,
            k,
            pools,
            realistic_underestimated_market.fees,
            realistic_underestimated_market,
            CalculationType.COST,
        )
        if N > 0
        else None
    )
    mc_above = agent._get_marginal_quote(
        N + 1,
        k,
        pools,
        realistic_underestimated_market.fees,
        realistic_underestimated_market,
        CalculationType.COST,
    )

    # 5) Assert that mc crosses 1.0 exactly at N
    assert (
        mc_below is None or mc_below < 1.0
    ), f"Marginal cost just below N should be <1 (got {mc_below:.4f})"
    # At N it should be right at or just under 1.0 (within a tiny epsilon)
    assert (
        pytest.approx(1.0, rel=1e-2) == mc_at_N
    ), f"Marginal cost at N should be ≈1 (got {mc_at_N:.4f})"
    # One more set should push marginal cost above 1
    assert mc_above > 1.0, f"Marginal cost above N should be >1 (got {mc_above:.4f})"

    # 6) Compute implied probabilities before & after
    #    m_i = k_i / x_i^2 ; then normalize
    def implied_probs(pools_dict: dict[OutcomeStr, float]) -> dict[OutcomeStr, float]:
        m = {o: k[o] / (pools_dict[o] ** 2) for o in pools_dict}
        total = sum(m.values())
        return {o: m[o] / total for o in m}

    # original pools and probs
    orig_sum = sum(probs.values())

    # new pools after buying N sets
    pools_after = {o: pools[o] - N for o in pools}
    new_probs = implied_probs(pools_after)
    new_sum = sum(new_probs.values())

    # 7) Assert that the sum has moved closer to 1
    assert abs(1.0 - new_sum) < abs(
        1.0 - orig_sum
    ), f"After buying, |1−{new_sum:.4f}| should be < |1−{orig_sum:.4f}|"


def test_max_sets_to_buy_with_high_liquidity(
    agent: DeployableFullSetCollectiveArbitrageAgent,
) -> None:
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
        fees=MarketFees(bet_proportion=0.01, absolute=0.02),  # Low fees
        collateral_token_usd_price=0.999,  # xDAI slightly below parity
    )

    mock_market.outcomes = realistic_market.outcomes
    mock_market.probabilities = probabilities
    mock_market.outcome_token_pool = realistic_market.outcome_token_pool
    mock_market.fees = realistic_market.fees
    mock_market.get_buy_token_amount = realistic_market.get_buy_token_amount
    mock_market.get_sell_value_of_outcome_token = (
        realistic_market.get_sell_value_of_outcome_token
    )
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    mock_market.get_liquidity = Mock(
        return_value=CollateralToken(1001001)
    )  # ~1M USD / 0.999

    N = agent._max_sets(mock_market, CalculationType.COST)

    # High liquidity should allow larger arbitrage quantities
    assert N > 10, "High liquidity should enable larger arbitrage trades"

    pools = {o: t.value for o, t in mock_market.outcome_token_pool.items()}
    probs = {o: float(p) for o, p in mock_market.probabilities.items()}
    # k_i = p_i * x_i^2
    k = {o: probs[o] * pools[o] ** 2 for o in pools}

    # 4) Compute the marginal cost at N, N - 1, and N + 1
    mc_at_N = agent._get_marginal_quote(
        N, k, pools, mock_market.fees, mock_market, CalculationType.COST
    )
    mc_below = (
        agent._get_marginal_quote(
            N - 1, k, pools, mock_market.fees, mock_market, CalculationType.COST
        )
        if N > 0
        else None
    )
    mc_above = agent._get_marginal_quote(
        N + 1, k, pools, mock_market.fees, mock_market, CalculationType.COST
    )

    # 5) Assert that mc crosses 1.0 exactly at N
    assert (
        mc_below is None or mc_below < 1.0
    ), f"Marginal cost just below N should be <1 (got {mc_below:.4f})"
    # At N it should be right at or just under 1.0 (within a tiny epsilon)
    assert (
        pytest.approx(1.0, rel=1e-2) == mc_at_N
    ), f"Marginal cost at N should be ≈1 (got {mc_at_N:.4f})"
    # One more set should push marginal cost above 1
    assert mc_above > 1.0, f"Marginal cost above N should be >1 (got {mc_above:.4f})"

    # 6) Compute implied probabilities before & after
    #    m_i = k_i / x_i^2 ; then normalize
    def implied_probs(pools_dict: dict[OutcomeStr, float]) -> dict[OutcomeStr, float]:
        m = {o: k[o] / (pools_dict[o] ** 2) for o in pools_dict}
        total = sum(m.values())
        return {o: m[o] / total for o in m}

    # original pools and probs
    orig_sum = sum(probs.values())

    # new pools after buying N sets
    pools_after = {o: pools[o] - N for o in pools}
    new_probs = implied_probs(pools_after)
    new_sum = sum(new_probs.values())

    # 7) Assert that the sum has moved closer to 1
    assert abs(1.0 - new_sum) < abs(
        1.0 - orig_sum
    ), f"After buying, |1−{new_sum:.4f}| should be < |1−{orig_sum:.4f}|"


def test_max_sets_to_buy_no_arbitrage_opportunity(
    agent: DeployableFullSetCollectiveArbitrageAgent,
) -> None:
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
    mock_market.get_sell_value_of_outcome_token = (
        realistic_market.get_sell_value_of_outcome_token
    )
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    mock_market.get_liquidity = Mock(
        return_value=CollateralToken(99502)
    )  # ~100k USD / 1.005

    N = agent._max_sets(mock_market, CalculationType.COST)

    # Should find no profitable arbitrage when probabilities sum to 1
    assert (
        N == 0
    ), "Should find no arbitrage opportunity when probabilities are balanced"


# --------------------------------------------------------------------------- #
# Tests for Overestimated Probabilities (sum > 1) - Mint and Sell           #
# --------------------------------------------------------------------------- #


def test_max_sets_to_mint_overestimated_simple(
    agent: DeployableFullSetCollectiveArbitrageAgent,
    realistic_overestimated_market: RealisticUniswapV3MockMarket,
) -> None:
    """
    Test max_sets_to_mint for overestimated probabilities.
    Should find optimal quantity where marginal sell value approaches 1.0.
    """  # Test the calculation
    N = agent._max_sets(realistic_overestimated_market, CalculationType.REVENUE)

    # With overestimated probabilities (1.20 total), we should find profitable opportunities
    assert N > 0, "Should find profitable mint-and-sell arbitrage opportunity"
    assert N <= 100, "Should be reasonable quantity given liquidity constraints"

    pools = (
        {
            o: t.value
            for o, t in realistic_overestimated_market.outcome_token_pool.items()
        }
        if realistic_overestimated_market.outcome_token_pool
        else {o: 0 for o in realistic_overestimated_market.outcomes}
    )
    probs = {
        o: float(p) for o, p in realistic_overestimated_market.probabilities.items()
    }
    # k_i = p_i * x_i^2
    k = {o: probs[o] * pools[o] ** 2 for o in pools}

    # Compute the marginal cost at N, N - 1, and N + 1
    rev_above = (
        agent._get_marginal_quote(
            N - 1,
            k,
            pools,
            realistic_overestimated_market.fees,
            realistic_overestimated_market,
            CalculationType.REVENUE,
        )
        if N > 0
        else None
    )
    rev_at = agent._get_marginal_quote(
        N,
        k,
        pools,
        realistic_overestimated_market.fees,
        realistic_overestimated_market,
        CalculationType.REVENUE,
    )
    rev_below = agent._get_marginal_quote(
        N + 1,
        k,
        pools,
        realistic_overestimated_market.fees,
        realistic_overestimated_market,
        CalculationType.REVENUE,
    )

    assert (
        rev_above is None or rev_above > 1.0
    ), f"Revenue just below N should be >1 (got {rev_above:.4f})"
    assert (
        pytest.approx(1.0, rel=1e-2) == rev_at
    ), f"Revenue at N should be ≈1 (got {rev_at:.4f})"
    assert rev_below < 1.0, f"Revenue above N should be <1 (got {rev_below:.4f})"

    # Compute implied probabilities before & after
    #    m_i = k_i / x_i^2 ; then normalize
    def implied_probs(pools_dict: dict[OutcomeStr, float]) -> dict[OutcomeStr, float]:
        m = {o: k[o] / (pools_dict[o] ** 2) for o in pools_dict}
        total = sum(m.values())
        return {o: m[o] / total for o in m}

    # original pools and probs
    orig_sum = sum(probs.values())

    # new pools after buying N sets
    pools_after = {o: pools[o] - N for o in pools}
    new_probs = implied_probs(pools_after)
    new_sum = sum(new_probs.values())

    # Assert that the sum has moved closer to 1
    assert abs(1.0 - new_sum) < abs(
        1.0 - orig_sum
    ), f"After buying, |1−{new_sum:.4f}| should be < |1−{orig_sum:.4f}|"


def test_max_sets_to_mint_with_price_impact(
    agent: DeployableFullSetCollectiveArbitrageAgent,
    realistic_overestimated_market: RealisticUniswapV3MockMarket,
) -> None:
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
        fees=MarketFees(bet_proportion=0.025, absolute=0.15),  # Higher fees
        collateral_token_usd_price=1.035,  # Higher yield sDAI
    )

    mock_market.outcomes = realistic_market.outcomes
    mock_market.probabilities = probabilities
    mock_market.outcome_token_pool = realistic_market.outcome_token_pool
    mock_market.fees = realistic_market.fees
    mock_market.get_buy_token_amount = realistic_market.get_buy_token_amount
    mock_market.get_sell_value_of_outcome_token = (
        realistic_market.get_sell_value_of_outcome_token
    )
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    mock_market.get_liquidity = Mock(
        return_value=CollateralToken(38647)
    )  # ~40k USD / 1.035

    N = agent._max_sets(mock_market, CalculationType.REVENUE)

    # Price impact should limit the arbitrage size despite large opportunity
    assert N > 0, "Should still find arbitrage despite price impact"
    assert N < 50, "Price impact should limit arbitrage size"

    pools = {o: t.value for o, t in mock_market.outcome_token_pool.items()}
    probs = {o: float(p) for o, p in mock_market.probabilities.items()}
    # k_i = p_i * x_i^2
    k = {o: probs[o] * pools[o] ** 2 for o in pools}

    # Compute the marginal cost at N, N - 1, and N + 1
    rev_above = (
        agent._get_marginal_quote(
            N - 1, k, pools, mock_market.fees, mock_market, CalculationType.REVENUE
        )
        if N > 0
        else None
    )
    rev_at = agent._get_marginal_quote(
        N, k, pools, mock_market.fees, mock_market, CalculationType.REVENUE
    )
    rev_below = agent._get_marginal_quote(
        N + 1, k, pools, mock_market.fees, mock_market, CalculationType.REVENUE
    )

    assert (
        rev_above is None or rev_above > 1.0
    ), f"Revenue just below N should be >1 (got {rev_above:.4f})"
    assert (
        pytest.approx(1.0, rel=1e-2) == rev_at
    ), f"Revenue at N should be ≈1 (got {rev_at:.4f})"
    assert rev_below < 1.0, f"Revenue above N should be <1 (got {rev_below:.4f})"

    # Compute implied probabilities before & after
    #    m_i = k_i / x_i^2 ; then normalize
    def implied_probs(pools_dict: dict[OutcomeStr, float]) -> dict[OutcomeStr, float]:
        m = {o: k[o] / (pools_dict[o] ** 2) for o in pools_dict}
        total = sum(m.values())
        return {o: m[o] / total for o in m}

    # original pools and probs
    orig_sum = sum(probs.values())

    # new pools after buying N sets
    pools_after = {o: pools[o] - N for o in pools}
    new_probs = implied_probs(pools_after)
    new_sum = sum(new_probs.values())

    # Assert that the sum has moved closer to 1
    assert abs(1.0 - new_sum) < abs(
        1.0 - orig_sum
    ), f"After buying, |1−{new_sum:.4f}| should be < |1−{orig_sum:.4f}|"


# --------------------------------------------------------------------------- #
# Edge Cases and Error Handling                                              #
# --------------------------------------------------------------------------- #


def test_max_sets_to_buy_zero_liquidity(
    agent: DeployableFullSetCollectiveArbitrageAgent,
) -> None:
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

    max_sets = agent._max_sets(mock_market, CalculationType.COST)

    # Should handle zero liquidity gracefully
    assert max_sets == 0, "Should return 0 for markets with no liquidity"


def test_max_sets_calculation_with_extreme_probabilities(
    agent: DeployableFullSetCollectiveArbitrageAgent,
) -> None:
    """Test with extreme probability imbalances."""
    # Very extreme underestimation
    probabilities = {
        OutcomeStr("Yes"): Probability(0.05),
        OutcomeStr("No"): Probability(0.10),  # 0.15 total - extreme underestimation
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
    mock_market.get_sell_value_of_outcome_token = (
        realistic_market.get_sell_value_of_outcome_token
    )
    mock_market.get_token_in_usd = realistic_market.get_token_in_usd
    mock_market.get_usd_in_token = realistic_market.get_usd_in_token
    mock_market.get_in_token = realistic_market.get_in_token
    mock_market.get_liquidity = Mock(
        return_value=CollateralToken(201005)
    )  # ~200k USD / 0.995

    N = agent._max_sets(mock_market, CalculationType.COST)

    # High liquidity should allow larger arbitrage quantities
    assert N > 10, "High liquidity should enable larger arbitrage trades"

    pools = {o: t.value for o, t in mock_market.outcome_token_pool.items()}
    probs = {o: float(p) for o, p in mock_market.probabilities.items()}
    # k_i = p_i * x_i^2
    k = {o: probs[o] * pools[o] ** 2 for o in pools}

    # 4) Compute the marginal cost at N, N - 1, and N + 1
    mc_at_N = agent._get_marginal_quote(
        N, k, pools, realistic_market.fees, realistic_market, CalculationType.COST
    )
    mc_below = (
        agent._get_marginal_quote(
            N - 1,
            k,
            pools,
            realistic_market.fees,
            realistic_market,
            CalculationType.COST,
        )
        if N > 0
        else None
    )
    mc_above = agent._get_marginal_quote(
        N + 1, k, pools, realistic_market.fees, realistic_market, CalculationType.COST
    )

    # 5) Assert that mc crosses 1.0 exactly at N
    assert (
        mc_below is None or mc_below < 1.0
    ), f"Marginal cost just below N should be <1 (got {mc_below:.4f})"
    # At N it should be right at or just under 1.0 (within a tiny epsilon)
    assert (
        pytest.approx(1.0, rel=1e-2) == mc_at_N
    ), f"Marginal cost at N should be ≈1 (got {mc_at_N:.4f})"
    # One more set should push marginal cost above 1
    assert mc_above > 1.0, f"Marginal cost above N should be >1 (got {mc_above:.4f})"

    # 6) Compute implied probabilities before & after
    #    m_i = k_i / x_i^2 ; then normalize
    def implied_probs(pools_dict: dict[OutcomeStr, float]) -> dict[OutcomeStr, float]:
        m = {o: k[o] / (pools_dict[o] ** 2) for o in pools_dict}
        total = sum(m.values())
        return {o: m[o] / total for o in m}

    # original pools and probs
    orig_sum = sum(probs.values())

    # new pools after buying N sets
    pools_after = {o: pools[o] - N for o in pools}
    new_probs = implied_probs(pools_after)
    new_sum = sum(new_probs.values())

    # 7) Assert that the sum has moved closer to 1
    assert abs(1.0 - new_sum) < abs(
        1.0 - orig_sum
    ), f"After buying, |1−{new_sum:.4f}| should be < |1−{orig_sum:.4f}|"


def test_high_fees_eating_up_marginal_revenue(
    agent: DeployableFullSetCollectiveArbitrageAgent,
    realistic_overestimated_market: RealisticUniswapV3MockMarket,
) -> None:
    """
    Test max_sets_to_mint for overestimated probabilities.
    Should find optimal quantity where marginal sell value approaches 1.0.
    """

    # Mock the precise pricing method for selling
    def mock_get_sell_price(
        market: SeerAgentMarket,
        outcome: OutcomeStr,
        tokens_to_sell_wei: Wei,
        from_address: ChecksumAddress,
    ) -> Wei:
        # Convert wei to tokens and get sell value
        tokens_to_sell = OutcomeToken(tokens_to_sell_wei.value / 10**18)
        collateral_received = market.get_sell_value_of_outcome_token(
            outcome, tokens_to_sell
        )
        return Wei(int(collateral_received.value * 10**18))

    realistic_overestimated_market.fees = MarketFees(bet_proportion=0.02, absolute=0.2)
    # Test the calculation
    N = agent._max_sets(realistic_overestimated_market, CalculationType.REVENUE)

    # With overestimated probabilities (1.20 total), we should find profitable opportunities
    assert N == 0, "Should find no arbitrage despite high fees"
