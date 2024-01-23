from functools import reduce
import numpy as np
import typing as t
from web3 import Web3

from prediction_market_agent.omen import get_market, omen_calculate_buy_amount
from prediction_market_agent.data_models.market_data_models import Market
from prediction_market_agent.tools.gnosis_rpc import GNOSIS_RPC_URL
from prediction_market_agent.tools.web3_utils import xdai_to_wei, wei_to_xdai
from prediction_market_agent.tools.types import xDai

OutcomeIndex = t.Literal[0, 1]


def get_market_moving_bet(
    market_address: str,
    target_p_yes: float,
    max_iters: int = 100,
    check_vs_contract: bool = False,  # Disable by default, as it's slow
    verbose: bool = False,
) -> t.Tuple[xDai, OutcomeIndex]:
    """
    Implements a binary search to determine the bet that will move the market's
    `p_yes` to that of the target.

    Consider a binary fixed-product market containing `x` and `y` tokens.
    A trader wishes to aquire `x` tokens by betting an amount `d0`.

    The calculation to determine the number of `x` tokens he acquires, denoted
    by `dx`, is:

    a_x * a_y = fixed_product
    na_x = a_x + d0
    na_y = a_y + d0
    na_x * na_y = new_product
    (na_x - dx) * na_y = fixed_product
    (na_x * na_y) - (dx * na_y) = fixed_product
    new_product - fixed_product = dx * na_y
    dx = (new_product - fixed_product) / na_y
    """
    web3 = Web3(Web3.HTTPProvider(GNOSIS_RPC_URL))
    market: Market = get_market(market_address)
    amounts = market.outcomeTokenAmounts
    prices = market.outcomeTokenMarginalPrices
    if len(amounts) != 2 or len(prices) != 2:
        raise ValueError("Only binary markets are supported.")

    fixed_product = reduce(lambda x, y: x * y, amounts, 1)
    assert np.isclose(float(sum(prices)), 1)
    current_p_yes = prices[0]
    bet_outcome_index = 0 if target_p_yes > current_p_yes else 1

    min_bet_amount = 0
    max_bet_amount = 100 * sum(amounts)  # TODO set a better upper bound

    # Binary search for the optimal bet amount
    for _ in range(max_iters):
        bet_amount = (min_bet_amount + max_bet_amount) / 2
        bet_amount_ = bet_amount * (xdai_to_wei(1) - market.fee) / xdai_to_wei(1)

        # Initial new amounts are old amounts + equal new amounts for each outcome
        amounts_diff = bet_amount_
        new_amounts = [amounts[i] + amounts_diff for i in range(len(amounts))]

        # Now give away tokens at `bet_outcome_index` to restore invariant
        new_product = reduce(lambda x, y: x * y, new_amounts, 1)
        dx = (new_product - fixed_product) / new_amounts[1 - bet_outcome_index]

        # Sanity check the number of tokens against the contract
        if check_vs_contract:
            expected_trade = omen_calculate_buy_amount(
                web3=web3,
                market=market,
                investment_amount=int(bet_amount),
                outcome_index=bet_outcome_index,
            )
            assert np.isclose(float(expected_trade), dx)

        new_amounts[bet_outcome_index] -= dx
        # Check that the invariant is restored
        assert np.isclose(
            reduce(lambda x, y: x * y, new_amounts, 1), float(fixed_product)
        )
        new_p_yes = new_amounts[1] / sum(new_amounts)
        if verbose:
            outcome = "'Yes'" if bet_outcome_index == 0 else "'No'"
            print(
                f"Target p_yes: {target_p_yes:.2f}, bet: {wei_to_xdai(bet_amount):.2f}xDai for {outcome}, new p_yes: {new_p_yes:.2f}"
            )
        if abs(target_p_yes - new_p_yes) < 0.01:
            return wei_to_xdai(bet_amount), bet_outcome_index
        elif new_p_yes > target_p_yes:
            if bet_outcome_index == 0:
                max_bet_amount = bet_amount
            else:
                min_bet_amount = bet_amount
        else:
            if bet_outcome_index == 0:
                min_bet_amount = bet_amount
            else:
                max_bet_amount = bet_amount


def _get_kelly_criterion_bet(
    x: int, y: int, p: float, c: float, b: int, f: float
) -> int:
    """
    Implments https://en.wikipedia.org/wiki/Kelly_criterion

    Taken from https://github.com/valory-xyz/trader/blob/main/strategies/kelly_criterion/kelly_criterion.py

    ```
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    ```

    x: Number of tokens in the selected outcome pool
    y: Number of tokens in the other outcome pool
    p: Probability of winning
    c: Confidence
    b: Bankroll
    f: Fee fraction
    """
    if b == 0:
        return 0
    numerator = (
        -4 * x**2 * y
        + b * y**2 * p * c * f
        + 2 * b * x * y * p * c * f
        + b * x**2 * p * c * f
        - 2 * b * y**2 * f
        - 2 * b * x * y * f
        + (
            (
                4 * x**2 * y
                - b * y**2 * p * c * f
                - 2 * b * x * y * p * c * f
                - b * x**2 * p * c * f
                + 2 * b * y**2 * f
                + 2 * b * x * y * f
            )
            ** 2
            - (
                4
                * (x**2 * f - y**2 * f)
                * (
                    -4 * b * x * y**2 * p * c
                    - 4 * b * x**2 * y * p * c
                    + 4 * b * x * y**2
                )
            )
        )
        ** (1 / 2)
    )
    denominator = 2 * (x**2 * f - y**2 * f)
    if denominator == 0:
        return 0
    kelly_bet_amount = numerator / denominator
    return int(kelly_bet_amount)


def get_kelly_criterion_bet(
    market_address: str,
    estimated_p_yes: float,
    max_bet: xDai,
) -> t.Tuple[xDai, OutcomeIndex]:
    market: Market = get_market(market_address)
    if len(market.outcomeTokenAmounts) != 2:
        raise ValueError("Only binary markets are supported.")

    current_p_yes = market.outcomeTokenMarginalPrices[0]
    outcome_index = 0 if estimated_p_yes > current_p_yes else 1
    estimated_p_win = estimated_p_yes if outcome_index == 0 else 1 - estimated_p_yes

    kelly_bet_wei = _get_kelly_criterion_bet(
        x=market.outcomeTokenAmounts[outcome_index],
        y=market.outcomeTokenAmounts[1 - outcome_index],
        p=estimated_p_win,
        c=1,  # confidence
        b=xdai_to_wei(max_bet),  # bankroll, or max bet, in Wei
        f=(xdai_to_wei(1) - market.fee) / xdai_to_wei(1),  # fee fraction
    )
    return wei_to_xdai(kelly_bet_wei), outcome_index


if __name__ == "__main__":
    market_address = "0xa3e47bb771074b33f2e279b9801341e9e0c9c6d7"
    est_p_yes = 0.1
    mov_bet = get_market_moving_bet(
        market_address=market_address,
        target_p_yes=est_p_yes,
        verbose=True,
    )
    kelly_bet = get_kelly_criterion_bet(
        market_address=market_address,
        estimated_p_yes=est_p_yes,
        max_bet=xDai(10),  # This significantly changes the outcome
    )

    market = get_market(market_address)

    def get_outcome_str(outcome_index: int) -> str:
        return market.get_outcome_str(outcome_index)

    get_market(market_address).get_outcome_str(0)
    print(f"Market moving bet: {mov_bet[0]:.2f} on {get_outcome_str(mov_bet[1])}")
    print(f"Kelly criterion bet: {kelly_bet[0]:.2f} on {get_outcome_str(kelly_bet[1])}")
