import pprint
import typing as t

from microchain import Function
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket

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
        # Get the 5 markets that are closing soonest
        markets: list[AgentMarket] = OmenAgentMarket.get_binary_markets(
            filter_by=FilterBy.OPEN,
            sort_by=SortBy.CLOSING_SOONEST,
            limit=5,
        )

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


class BuyYes(Function):
    @property
    def description(self) -> str:
        return "Use this function to buy yes outcome tokens of a prediction market. The second parameter specifies how much $ you spend."

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return ["Will Joe Biden get reelected in 2024?", 2]

    def __call__(self, market: str, amount: int) -> str:
        global balance
        if amount > balance:
            return (
                f"Your balance of {balance} $ is not large enough to spend {amount} $."
            )

        balance -= amount
        return "Bought " + str(amount * 2) + " yes outcome token of: " + market


class BuyNo(Function):
    @property
    def description(self) -> str:
        return "Use this function to buy no outcome tokens of a prdiction market. The second parameter specifies how much $ you spend."

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return ["Will Joe Biden get reelected in 2024?", 4]

    def __call__(self, market: str, amount: int) -> str:
        global balance
        if amount > balance:
            return (
                f"Your balance of {balance} $ is not large enough to spend {amount} $."
            )

        balance -= amount
        return "Bought " + str(amount * 2) + " no outcome token of: " + market


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
]
