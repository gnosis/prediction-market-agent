import typing as t

from microchain import Function
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import Currency, TokenAmount
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket

from prediction_market_agent.agents.microchain_agent.utils import (
    MechResult,
    MechTool,
    MicrochainAPIKeys,
    MicroMarket,
    get_balance,
    get_binary_markets,
    get_boolean_outcome,
    get_example_market_id,
    get_no_outcome,
    get_yes_outcome,
    mech_request,
    mech_request_local,
)


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

    @property
    def currency(self) -> Currency:
        return self.market_type.market_class.currency


class GetMarkets(MarketFunction):
    @property
    def description(self) -> str:
        return (
            "Use this function to get a list of predction market questions, "
            "and the corresponding market IDs"
        )

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> list[str]:
        return [
            str(MicroMarket.from_agent_market(m))
            for m in get_binary_markets(market_type=self.market_type)
        ]


class GetMarketProbability(MarketFunction):
    @property
    def description(self) -> str:
        return (
            f"Use this function to get the probability of a 'Yes' outcome for "
            f"a binary prediction market. This is equivalent to the price of "
            f"the 'Yes' token in {self.currency}. Pass in the market id as a "
            f"string."
        )

    @property
    def example_args(self) -> list[str]:
        return [get_example_market_id(self.market_type)]

    def __call__(self, market_id: str) -> list[str]:
        return [
            str(self.market_type.market_class.get_binary_market(id=market_id).p_yes)
        ]


class PredictProbabilityForQuestionBase(MarketFunction):
    def __init__(
        self,
        mech_request: t.Callable[[str, MechTool], MechResult],
        market_type: MarketType,
        mech_tool: MechTool = MechTool.PREDICTION_ONLINE,
    ) -> None:
        self.mech_tool = mech_tool
        self.mech_request = mech_request
        self._description = (
            "Use this function to research perform research and predict the "
            "probability of an event occuring. Returns the probability. The "
            "one parameter is the market id of the prediction market you want "
            "to predict the probability of."
        )
        super().__init__(market_type=market_type)

    @property
    def example_args(self) -> list[str]:
        return [get_example_market_id(self.market_type)]

    def __call__(self, market_id: str) -> str:
        question = self.market_type.market_class.get_binary_market(
            id=market_id
        ).question
        result: MechResult = self.mech_request(question, self.mech_tool)
        return str(result.p_yes)


class PredictProbabilityForQuestionRemote(PredictProbabilityForQuestionBase):
    def __init__(
        self,
        market_type: MarketType,
        mech_tool: MechTool = MechTool.PREDICTION_ONLINE,
    ) -> None:
        self.mech_tool = mech_tool
        super().__init__(market_type=market_type, mech_request=mech_request)

    @property
    def description(self) -> str:
        return self._description + " Note, this costs money to run."

    def __call__(self, market_id: str) -> str:
        # 0.01 xDai is hardcoded cost for an interaction with the mech-client
        MECH_CALL_XDAI_LIMIT = 0.011
        account_balance = float(get_balance(market_type=self.market_type).amount)
        if account_balance < MECH_CALL_XDAI_LIMIT:
            return (
                f"Your balance of {self.currency} ({account_balance}) is not "
                f"large enough to make a mech call (min required "
                f"{MECH_CALL_XDAI_LIMIT})."
            )
        return super().__call__(market_id)


class PredictProbabilityForQuestionLocal(PredictProbabilityForQuestionBase):
    def __init__(
        self,
        market_type: MarketType,
        mech_tool: MechTool = MechTool.PREDICTION_ONLINE,
    ) -> None:
        self.mech_tool = mech_tool
        super().__init__(market_type=market_type, mech_request=mech_request_local)

    @property
    def description(self) -> str:
        return self._description


class BuyTokens(MarketFunction):
    def __init__(self, market_type: MarketType, outcome: str):
        self.outcome = outcome
        self.outcome_bool = get_boolean_outcome(
            outcome=self.outcome, market_type=market_type
        )
        self.user_address = MicrochainAPIKeys().bet_from_address
        super().__init__(market_type=market_type)

    @property
    def description(self) -> str:
        return (
            f"Use this function to buy {self.outcome} outcome tokens of a "
            f"prediction market. The first parameter is the market id. The "
            f"second parameter specifies how much {self.currency} you spend."
        )

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return [get_example_market_id(self.market_type), 2.3]

    def __call__(self, market_id: str, amount: float) -> str:
        market: AgentMarket = self.market_type.market_class.get_binary_market(market_id)
        account_balance = float(get_balance(market_type=self.market_type).amount)
        if account_balance < amount:
            return (
                f"Your balance of {self.currency} ({account_balance}) is not "
                f"large enough to buy {amount} tokens."
            )

        before_balance = market.get_token_balance(
            user_id=self.user_address,
            outcome=self.outcome,
        )
        market.buy_tokens(
            outcome=self.outcome_bool,
            amount=TokenAmount(amount=amount, currency=self.currency),
        )
        after_balance = market.get_token_balance(
            user_id=self.user_address,
            outcome=self.outcome,
        )
        tokens = float(after_balance.amount - before_balance.amount)
        return f"Bought {tokens} {self.outcome} outcome tokens of market with id: {market_id}"


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


class SellTokens(MarketFunction):
    def __init__(self, market_type: MarketType, outcome: str):
        self.outcome = outcome
        self.outcome_bool = get_boolean_outcome(
            outcome=self.outcome,
            market_type=market_type,
        )
        self.user_address = MicrochainAPIKeys().bet_from_address
        super().__init__(market_type=market_type)

    @property
    def description(self) -> str:
        return (
            f"Use this function to sell {self.outcome} outcome tokens of a "
            f"prediction market. The first parameter is the market id. The "
            f"second parameter specifies the value of tokens to sell in "
            f"{self.currency}."
        )

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return [get_example_market_id(self.market_type), 2.3]

    def __call__(self, market_id: str, amount: float) -> str:
        market: AgentMarket = self.market_type.market_class.get_binary_market(market_id)
        before_balance = market.get_token_balance(
            user_id=self.user_address,
            outcome=self.outcome,
        )

        market.sell_tokens(
            outcome=self.outcome_bool,
            amount=TokenAmount(amount=amount, currency=self.currency),
        )

        after_balance = market.get_token_balance(
            user_id=self.user_address,
            outcome=self.outcome,
        )
        tokens = float(before_balance.amount - after_balance.amount)
        return f"Sold {tokens} {self.outcome} outcome tokens of market with id: {market_id}"


class SellYes(SellTokens):
    def __init__(self, market_type: MarketType) -> None:
        super().__init__(
            market_type=market_type, outcome=get_yes_outcome(market_type=market_type)
        )


class SellNo(SellTokens):
    def __init__(self, market_type: MarketType) -> None:
        super().__init__(
            market_type=market_type, outcome=get_no_outcome(market_type=market_type)
        )


class RedeemWinningBets(MarketFunction):
    @property
    def description(self) -> str:
        # ToDo - implement
        return "Use this function summarize your learnings and save them so that you can access them later."

    @property
    def example_args(self) -> list[str]:
        # ToDo - implement
        return [
            "Today I learned that I need to check my balance fore making decisions about how much to invest."
        ]

    def __call__(self, market_id: str) -> str:
        # ToDo - implement
        # ToDo - Implement functions in PMAT (redeemable by, redeem_positions) in general agent, release new version
        #  market_redeemable_by, if yes, then call redeem_positions
        keys = MicrochainAPIKeys()
        # ToDo - Make it AgentMarket instead of Omen after changes above were implemented
        market: AgentMarket = self.market_type.market_class.get_binary_market(market_id)
        omen_agent_market = OmenAgentMarket()
        user_address = MicrochainAPIKeys().bet_from_address
        if omen_agent_market.market_redeemable_by(user_address):
            omen_agent_market.redeem_positions(keys.bet_from_private_key)


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
        return summary


class GetBalance(MarketFunction):
    @property
    def description(self) -> str:
        return (
            f"Use this function to fetch your balance, given in {self.currency} units."
        )

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> float:
        return get_balance(market_type=self.market_type).amount


class GetPositions(MarketFunction):
    def __init__(self, market_type: MarketType) -> None:
        self.user_address = MicrochainAPIKeys().bet_from_address
        super().__init__(market_type=market_type)

    @property
    def description(self) -> str:
        return (
            "Use this function to fetch the live markets where you have "
            "previously bet, and the token amounts you hold for each outcome."
        )

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> list[str]:
        self.user_address = MicrochainAPIKeys().bet_from_address
        positions = self.market_type.market_class.get_positions(
            user_id=self.user_address
        )
        return [str(position) for position in positions]


MISC_FUNCTIONS = [
    Sum,
    Product,
    # SummarizeLearning,
]

# Functions that interact with the prediction markets
MARKET_FUNCTIONS: list[type[MarketFunction]] = [
    GetMarkets,
    GetMarketProbability,
    # PredictProbabilityForQuestionRemote, # Quite slow, use local version for now
    PredictProbabilityForQuestionLocal,
    GetBalance,
    BuyYes,
    BuyNo,
    SellYes,
    SellNo,
    GetPositions,
]
