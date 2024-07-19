import typing as t
from datetime import timedelta

from microchain import Function
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import (
    Currency,
    ResolvedBet,
    TokenAmount,
)
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.tools.utils import utcnow
from prediction_prophet.benchmark.agents import (
    _make_prediction as prophet_make_prediction,
)
from prediction_prophet.functions.research import research as prophet_research

from prediction_market_agent.agents.microchain_agent.utils import (
    MicroMarket,
    get_balance,
    get_binary_markets,
    get_boolean_outcome,
    get_example_market_id,
    get_no_outcome,
    get_yes_outcome,
)
from prediction_market_agent.tools.mech.utils import MechResponse, MechTool
from prediction_market_agent.utils import APIKeys


class MarketFunction(Function):
    def __init__(self, market_type: MarketType, keys: APIKeys) -> None:
        self.keys = keys
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
            str(
                self.market_type.market_class.get_binary_market(
                    id=market_id
                ).current_p_yes
            )
        ]


class PredictProbabilityForQuestionBase(MarketFunction):
    def __init__(
        self,
        market_type: MarketType,
        keys: APIKeys,
    ) -> None:
        super().__init__(market_type=market_type, keys=keys)
        self._description = (
            "Use this function to research perform research and predict the "
            "probability of an event occuring. Returns the probability. The "
            "one parameter is the market id of the prediction market you want "
            "to predict the probability of."
        )

    @property
    def example_args(self) -> list[str]:
        return [get_example_market_id(self.market_type)]


class PredictProbabilityForQuestion(PredictProbabilityForQuestionBase):
    """
    Uses the prediction_prophet library to make a prediction.
    """

    def __init__(
        self,
        market_type: MarketType,
        keys: APIKeys,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        self.model = model
        super().__init__(market_type=market_type, keys=keys)

    @property
    def description(self) -> str:
        return self._description

    def __call__(self, market_id: str) -> str:
        question = self.market_type.market_class.get_binary_market(
            id=market_id
        ).question
        research = prophet_research(
            goal=question,
            use_summaries=False,
            model=self.model,
            openai_api_key=self.keys.openai_api_key,
            tavily_api_key=self.keys.tavily_api_key,
        )
        prediction = prophet_make_prediction(
            market_question=question,
            additional_information=research,
            engine=self.model,
            temperature=0,
            api_key=self.keys.openai_api_key,
        )
        if prediction.outcome_prediction is None:
            raise ValueError("Failed to make a prediction.")

        return str(prediction.outcome_prediction.p_yes)


class PredictProbabilityForQuestionMech(PredictProbabilityForQuestionBase):
    """
    Uses the mech-client to make a prediction. Useability issues:
    https://github.com/gnosis/prediction-market-agent/issues/327
    """

    def __init__(
        self,
        market_type: MarketType,
        keys: APIKeys,
        mech_tool: MechTool = MechTool.PREDICTION_ONLINE,
    ) -> None:
        self.mech_tool = mech_tool
        super().__init__(market_type=market_type, keys=keys)

    @property
    def description(self) -> str:
        return self._description + " Note, this costs money to run."

    def __call__(self, market_id: str) -> str:
        # 0.01 xDai is hardcoded cost for an interaction with the mech-client
        MECH_CALL_XDAI_LIMIT = 0.011
        account_balance = float(
            get_balance(self.keys, market_type=self.market_type).amount
        )
        if account_balance < MECH_CALL_XDAI_LIMIT:
            return (
                f"Your balance of {self.currency} ({account_balance}) is not "
                f"large enough to make a mech call (min required "
                f"{MECH_CALL_XDAI_LIMIT})."
            )

        question = self.market_type.market_class.get_binary_market(
            id=market_id
        ).question
        response: MechResponse = self.mech_request(question, self.mech_tool)
        return str(response.p_yes)


class BuyTokens(MarketFunction):
    def __init__(self, market_type: MarketType, outcome: str, keys: APIKeys):
        super().__init__(market_type=market_type, keys=keys)
        self.outcome = outcome
        self.outcome_bool = get_boolean_outcome(
            outcome=self.outcome, market_type=market_type
        )
        self.user_address = self.keys.bet_from_address

        # Prevent the agent from spending recklessly!
        self.MAX_AMOUNT = 0.1 if market_type == MarketType.OMEN else 1.0

    @property
    def description(self) -> str:
        return (
            f"Use this function to buy {self.outcome} outcome tokens of a "
            f"prediction market. The first parameter is the market id. The "
            f"second parameter specifies how much {self.currency} you spend."
            f"This is capped at {self.MAX_AMOUNT}{self.currency}."
        )

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return [get_example_market_id(self.market_type), 2.3]

    def __call__(self, market_id: str, amount: float) -> str:
        if amount > self.MAX_AMOUNT:
            return f"Failed. Bet amount {amount} cannot exceed {self.MAX_AMOUNT} {self.currency}."

        account_balance = float(
            get_balance(self.keys, market_type=self.market_type).amount
        )
        if account_balance < amount:
            return (
                f"Your balance of {self.currency} ({account_balance}) is not "
                f"large enough to buy {amount} {self.currency} worth of tokens."
            )

        market: AgentMarket = self.market_type.market_class.get_binary_market(market_id)
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
    def __init__(self, market_type: MarketType, keys: APIKeys) -> None:
        super().__init__(
            market_type=market_type,
            keys=keys,
            outcome=get_yes_outcome(market_type=market_type),
        )


class BuyNo(BuyTokens):
    def __init__(self, market_type: MarketType, keys: APIKeys) -> None:
        super().__init__(
            market_type=market_type,
            keys=keys,
            outcome=get_no_outcome(market_type=market_type),
        )


class SellTokens(MarketFunction):
    def __init__(self, market_type: MarketType, outcome: str, keys: APIKeys):
        super().__init__(market_type=market_type, keys=keys)
        self.outcome = outcome
        self.outcome_bool = get_boolean_outcome(
            outcome=self.outcome,
            market_type=market_type,
        )
        self.user_address = self.keys.bet_from_address

    @property
    def description(self) -> str:
        return (
            f"Use this function to sell {self.outcome} outcome tokens of a "
            f"prediction market. The first parameter is the market id. The "
            f"second parameter specifies the (float) number of tokens to sell. "
            f"You can use `GetLiquidPositions` to see your current token "
            f"holdings."
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
    def __init__(self, market_type: MarketType, keys: APIKeys) -> None:
        super().__init__(
            market_type=market_type,
            keys=keys,
            outcome=get_yes_outcome(market_type=market_type),
        )


class SellNo(SellTokens):
    def __init__(self, market_type: MarketType, keys: APIKeys) -> None:
        super().__init__(
            market_type=market_type,
            keys=keys,
            outcome=get_no_outcome(market_type=market_type),
        )


class GetBalance(MarketFunction):
    def __init__(self, market_type: MarketType, keys: APIKeys) -> None:
        super().__init__(market_type=market_type, keys=keys)

    @property
    def description(self) -> str:
        return (
            f"Use this function to fetch your balance, given in {self.currency} units."
        )

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> float:
        return get_balance(self.keys, market_type=self.market_type).amount


class GetLiquidPositions(MarketFunction):
    def __init__(self, market_type: MarketType, keys: APIKeys) -> None:
        super().__init__(market_type=market_type, keys=keys)
        self.user_address = self.keys.bet_from_address

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
        self.user_address = self.keys.bet_from_address
        positions = self.market_type.market_class.get_positions(
            user_id=self.user_address,
            liquid_only=True,
            larger_than=1e-4,  # Ignore very small positions
        )
        return [str(position) for position in positions]


class GetResolvedBetsWithOutcomes(MarketFunction):
    def __init__(self, market_type: MarketType, keys: APIKeys) -> None:
        super().__init__(market_type=market_type, keys=keys)
        self.user_address = self.keys.bet_from_address

    @property
    def description(self) -> str:
        return (
            "Use this function to fetch the outcomes of previous bets you have placed."
            f"Pass in the number of days (as an integer) in the past you want to look back from the current start date `{utcnow()}`."
        )

    @property
    def example_args(self) -> list[int]:
        return [7]

    def __call__(self, n_days: int = 7) -> list[ResolvedBet]:
        # We look back a standard interval as a rule-of-thumb for now.
        start_time = utcnow() - timedelta(days=n_days)
        return self.market_type.market_class.get_resolved_bets_made_since(
            better_address=self.user_address, start_time=start_time, end_time=None
        )


# Functions that interact with the prediction markets
MARKET_FUNCTIONS: list[type[MarketFunction]] = [
    GetMarkets,
    GetMarketProbability,
    PredictProbabilityForQuestion,
    GetBalance,
    BuyYes,
    BuyNo,
    SellYes,
    SellNo,
    GetLiquidPositions,
    GetResolvedBetsWithOutcomes,
]
