import pprint
import typing as t
from decimal import Decimal
from typing import cast

from eth_utils import to_checksum_address
from microchain import Function
from prediction_market_agent_tooling.config import APIKeys
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import BetAmount, Currency
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.data_models import (
    OmenUserPosition,
    get_boolean_outcome,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    OmenSubgraphHandler,
)
from prediction_market_agent_tooling.tools.balances import get_balances

from prediction_market_agent.agents.microchain_agent.utils import (
    MicroMarket,
    get_binary_market_from_question,
    get_binary_markets,
    get_market_token_balance,
    get_no_outcome,
    get_yes_outcome,
)

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


class MarketFunction(Function):
    def __init__(self, market_type: MarketType) -> None:
        self.market_type = market_type
        super().__init__()


class GetMarkets(MarketFunction):
    @property
    def description(self) -> str:
        return "Use this function to get a list of predction markets and the current yes prices"

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> list[str]:
        return [
            str(MicroMarket.from_agent_market(m))
            for m in get_binary_markets(market_type=self.market_type)
        ]


class GetPropabilityForQuestion(MarketFunction):
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


class GetBalance(MarketFunction):
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


class BuyTokens(MarketFunction):
    def __init__(self, market_type: MarketType, outcome: str):
        self.outcome = outcome
        self.user_address = APIKeys().bet_from_address
        super().__init__(market_type=market_type)

    @property
    def description(self) -> str:
        return f"Use this function to buy {self.outcome} outcome tokens of a prediction market. The second parameter specifies how much $ you spend."

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return ["Will Joe Biden get reelected in 2024?", 2.3]

    def __call__(self, market: str, amount: float) -> str:
        outcome_bool = get_boolean_outcome(self.outcome)

        market_obj: AgentMarket = get_binary_market_from_question(
            market=market, market_type=self.market_type
        )
        market_obj = cast(
            OmenAgentMarket, market_obj
        )  # TODO fix with 0.10.0 PMAT release
        outcome_index = market_obj.get_outcome_index(self.outcome)
        market_index_set = outcome_index + 1

        before_balance = get_market_token_balance(
            user_address=self.user_address,
            market_condition_id=market_obj.condition.id,
            market_index_set=market_index_set,
        )
        market_obj.place_bet(
            outcome_bool, BetAmount(amount=Decimal(amount), currency=Currency.xDai)
        )
        tokens = (
            get_market_token_balance(
                user_address=self.user_address,
                market_condition_id=market_obj.condition.id,
                market_index_set=market_index_set,
            )
            - before_balance
        )
        return f"Bought {tokens} {self.outcome} outcome tokens of: {market}"


class BuyYes(BuyTokens):
    def __init__(self, market_type: MarketType) -> None:
        super().__init__(
            market_type=market_type, outcome=get_yes_outcome(market_type=market_type)
        )


class BuyNo(BuyTokens):
    def __init__(self, market_type: MarketType) -> None:
        super().__init__(
            market_type=market_type, outcome=get_no_outcome(market_type=market_type)
        )


class SellYes(MarketFunction):
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


class SellNo(MarketFunction):
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


class GetWalletBalance(MarketFunction):
    @property
    def description(self) -> str:
        return "Use this function to fetch your balance, given in xDAI units."

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self, user_address: str) -> Decimal:
        # We focus solely on xDAI balance for now to avoid the agent having to wrap/unwrap xDAI.
        user_address_checksummed = to_checksum_address(user_address)
        balance = get_balances(user_address_checksummed)
        return balance.xdai


class GetUserPositions(MarketFunction):
    @property
    def description(self) -> str:
        return (
            "Use this function to fetch the markets where the user has previously bet."
        )

    @property
    def example_args(self) -> list[str]:
        return ["0x2DD9f5678484C1F59F97eD334725858b938B4102"]

    def __call__(self, user_address: str) -> list[OmenUserPosition]:
        return OmenSubgraphHandler().get_user_positions(
            better_address=to_checksum_address(user_address)
        )


MISC_FUNCTIONS = [
    Sum,
    Product,
    SummarizeLearning,
]

# Functions that interact with the prediction markets
MARKET_FUNCTIONS: list[type[MarketFunction]] = [
    GetMarkets,
    GetPropabilityForQuestion,
    GetBalance,
    BuyYes,
    BuyNo,
    SellYes,
    SellNo,
    GetWalletBalance,
    GetUserPositions,
]
