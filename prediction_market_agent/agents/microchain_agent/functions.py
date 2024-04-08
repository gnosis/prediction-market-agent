import os
import pprint
import typing as t
from decimal import Decimal

from eth_typing import ChecksumAddress, HexAddress, HexStr
from microchain import Function
from prediction_market_agent_tooling.gtypes import xDai
from prediction_market_agent_tooling.markets.data_models import BetAmount, Currency
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.tools.balances import get_balances
from prediction_market_agent_tooling.tools.web3_utils import private_key_to_public_key
from pydantic import SecretStr

from prediction_market_agent.agents.microchain_agent.tools import (
    get_omen_binary_market_from_question,
    get_omen_binary_markets,
    get_omen_market_token_balance, address_to_checksum_address,
)

#PRIVATE_KEY = os.getenv("BET_FROM_PRIVATE_KEY")
#PUBLIC_KEY = private_key_to_public_key(SecretStr(PRIVATE_KEY))

balance = 50
outcomeTokens = {}
outcomeTokens["Will Joe Biden get reelected in 2024?"] = {"yes": 0, "no": 0}
outcomeTokens["Will Bitcoin hit 100k in 2024?"] = {"yes": 0, "no": 0}


class Sum(Function):
    @property
    def description(self) -> str:
        return "Use this function to compute the sum of two numbers"

    @property
    def example_args(self) -> list[float]:
        return [2, 2]

    def __call__(self, a: float, b: float) -> float:
        return a + b


class Product(Function):
    @property
    def description(self) -> str:
        return "Use this function to compute the product of two numbers"

    @property
    def example_args(self) -> list[float]:
        return [2, 2]

    def __call__(self, a: float, b: float) -> float:
        return a * b


class GetMarkets(Function):
    @property
    def description(self) -> str:
        return "Use this function to get a list of predction markets and the current yes prices"

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> list[str]:
        markets: list[OmenAgentMarket] = get_omen_binary_markets()
        market_questions_and_prices = []
        for market in markets:
            market_questions_and_prices.append(market.question)
            market_questions_and_prices.append(str(market.p_yes))
        return market_questions_and_prices


class GetPropabilityForQuestion(Function):
    @property
    def description(self) -> str:
        return "Use this function to research the probability of an event occuring"

    @property
    def example_args(self) -> list[str]:
        return ["Will Joe Biden get reelected in 2024?"]

    def __call__(self, a: str) -> float:
        if a == "Will Joe Biden get reelected in 2024?":
            return 0.41
        if a == "Will Bitcoin hit 100k in 2024?":
            return 0.22

        return 0.0


class GetBalance(Function):
    @property
    def description(self) -> str:
        return "Use this function to get your own balance in $"

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> float:
        print(f"Your balance is: {balance} and ")
        pprint.pprint(outcomeTokens)
        return balance


class BuyTokens(Function):
    def __init__(self, outcome: str):
        self.outcome = outcome
        super().__init__()

    @property
    def description(self) -> str:
        return f"Use this function to buy {self.outcome} outcome tokens of a prediction market. The second parameter specifies how much $ you spend."

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return ["Will Joe Biden get reelected in 2024?", 2.3]

    def __call__(self, market: str, amount: float) -> str:
        if self.outcome == "yes":
            outcome_bool = True
        elif self.outcome == "no":
            outcome_bool = False
        else:
            raise ValueError(f"Invalid outcome: {self.outcome}")

        market_obj: OmenAgentMarket = get_omen_binary_market_from_question(market)
        before_balance = get_omen_market_token_balance(
            market=market_obj, outcome=outcome_bool
        )
        market_obj.place_bet(
            outcome_bool, BetAmount(amount=amount, currency=Currency.xDai)
        )
        tokens = (
                get_omen_market_token_balance(market=market_obj, outcome=outcome_bool)
                - before_balance
        )
        return f"Bought {tokens} {self.outcome} outcome tokens of: {market}"


class BuyYes(BuyTokens):
    def __init__(self):
        super().__init__("yes")


class BuyNo(BuyTokens):
    def __init__(self):
        super().__init__("no")


class SellYes(Function):
    @property
    def description(self) -> str:
        return "Use this function to sell yes outcome tokens of a prediction market. The second parameter specifies how much tokens you sell."

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return ["Will Joe Biden get reelected in 2024?", 2]

    def __call__(self, market: str, amount: int) -> str:
        global outcomeTokens
        if amount > outcomeTokens[market]["yes"]:
            return f"Your balance of {outcomeTokens[market]['yes']} yes outcome tokens is not large enough to sell {amount}."

        outcomeTokens[market]["yes"] -= amount
        return "Sold " + str(amount) + " yes outcome token of: " + market


class SellNo(Function):
    @property
    def description(self) -> str:
        return "Use this function to sell no outcome tokens of a prdiction market. The second parameter specifies how much tokens you sell."

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return ["Will Joe Biden get reelected in 2024?", 4]

    def __call__(self, market: str, amount: int) -> str:
        global outcomeTokens
        if amount > outcomeTokens[market]["no"]:
            return f"Your balance of {outcomeTokens[market]['no']} no outcome tokens is not large enough to sell {amount}."

        outcomeTokens[market]["no"] -= amount
        return "Sold " + str(amount) + " no outcome token of: " + market


class BalanceToOutcomes(Function):
    @property
    def description(self) -> str:
        return "Use this function to convert your balance into equal units of 'yes' and 'no' outcome tokens. The function takes the amount of balance as the argument."

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return ["Will Joe Biden get reelected in 2024?", 50]

    def __call__(self, market: str, amount: int) -> str:
        global balance
        global outcomeTokens
        outcomeTokens[market]["yes"] += amount
        outcomeTokens[market]["no"] += amount
        balance -= amount
        return f"Converted {amount} units of balance into {amount} 'yes' outcome tokens and {amount} 'no' outcome tokens. Remaining balance is {balance}."


class SummarizeLearning(Function):
    @property
    def description(self) -> str:
        return "Use this function summarize your learnings and save them so that you can access them later."

    @property
    def example_args(self) -> list[str]:
        return [
            "Today I learned that I need to check my balance fore making decisions about how much to invest."
        ]

    def __call__(self, summary: str) -> str:
        # print(summary)
        # pprint.pprint(outcomeTokens)
        return summary


class GetWalletBalance(Function):
    @property
    def description(self) -> str:
        return "Use this function to fetch the balance of a user, given in xDAI units."

    @property
    def example_args(self) -> list[str]:
        return ["0x2DD9f5678484C1F59F97eD334725858b938B4102"]

    def __call__(self, user_address: str) -> Decimal:
        # We focus solely on xDAI balance for now to avoid the agent having to wrap/unwrap xDAI.
        user_address_checksummed = address_to_checksum_address(user_address)
        balance = get_balances(user_address_checksummed)
        return balance.xdai


ALL_FUNCTIONS = [
    Sum,
    Product,
    GetMarkets,
    GetPropabilityForQuestion,
    GetBalance,
    BuyYes,
    BuyNo,
    SellYes,
    SellNo,
    # BalanceToOutcomes,
    SummarizeLearning,
    GetWalletBalance
]
