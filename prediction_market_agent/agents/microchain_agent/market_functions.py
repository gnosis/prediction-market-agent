import typing as t
from datetime import timedelta

from microchain import Function
from prediction_market_agent_tooling.gtypes import USD, OutcomeStr, OutcomeToken, xDai
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ResolvedBet
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.markets.omen.omen import (
    send_keeping_token_to_eoa_xdai,
)
from prediction_market_agent_tooling.tools.betting_strategies.kelly_criterion import (
    get_kelly_bet_simplified,
)
from prediction_market_agent_tooling.tools.tokens.usd import get_usd_in_xdai
from prediction_market_agent_tooling.tools.utils import utcnow
from pydantic_ai import Agent as Agent
from pydantic_ai.models import KnownModelName
from pydantic_ai.settings import ModelSettings

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
from prediction_market_agent.tools.prediction_prophet.research import (
    prophet_make_prediction,
    prophet_research,
)
from prediction_market_agent.utils import DEFAULT_OPENAI_MODEL, APIKeys

OMEN_MIN_FEE_BALANCE = xDai(0.01)
MULTIPLIER = 2


class MarketFunction(Function):
    def __init__(self, market_type: MarketType, keys: APIKeys) -> None:
        self.keys = keys
        self.market_type = market_type
        super().__init__()


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
            f"the 'Yes' token in market's currency. Pass in the market id as a "
            f"string."
        )

    @property
    def example_args(self) -> list[str]:
        return [get_example_market_id(self.market_type)]

    def __call__(self, market_id: str) -> list[str]:
        market = self.market_type.market_class.get_binary_market(id=market_id)
        market_p_yes = market.p_yes
        return [str(market_p_yes)]


class PredictProbabilityForQuestionBase(MarketFunction):
    def __init__(
        self,
        market_type: MarketType,
        keys: APIKeys,
    ) -> None:
        super().__init__(market_type=market_type, keys=keys)
        self._description = (
            "Use this function to research perform research and predict the "
            "probability of an event occurring. Returns the probability. The "
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
        model: KnownModelName = DEFAULT_OPENAI_MODEL,
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
            agent=Agent(self.model, model_settings=ModelSettings(temperature=0.7)),
            openai_api_key=self.keys.openai_api_key,
            tavily_api_key=self.keys.tavily_api_key,
        )
        prediction = prophet_make_prediction(
            market_question=question,
            additional_information=research.report,
            agent=Agent(self.model, model_settings=ModelSettings(temperature=0)),
        )
        if prediction is None:
            raise ValueError("Failed to make a prediction.")

        return str(prediction.p_yes)


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
        mech_call_xdai_limit = xDai(0.011)
        account_balance = get_balance(self.keys, market_type=self.market_type)
        if get_usd_in_xdai(account_balance) < mech_call_xdai_limit:
            return (
                f"Your balance of {account_balance} is not "
                f"large enough to make a mech call (min required "
                f"{mech_call_xdai_limit})."
            )

        question = self.market_type.market_class.get_binary_market(
            id=market_id
        ).question
        response: MechResponse = self.mech_request(question, self.mech_tool)
        return str(response.p_yes)


class BuyTokens(MarketFunction):
    def __init__(self, market_type: MarketType, outcome: OutcomeStr, keys: APIKeys):
        super().__init__(market_type=market_type, keys=keys)
        self.outcome = outcome
        self.outcome_bool = get_boolean_outcome(
            outcome=self.outcome, market_type=market_type
        )
        self.user_address = self.keys.bet_from_address

    @property
    def description(self) -> str:
        return (
            f"Use this function to buy {self.outcome} outcome tokens of a "
            f"prediction market. The first parameter is the market id. The "
            f"second parameter specifies how much you spend."
        )

    @property
    def example_args(self) -> list[t.Union[str, float]]:
        return [get_example_market_id(self.market_type), 2.3]

    def __call__(self, market_id: str, amount_usd: float) -> str:
        amount = USD(amount_usd)
        account_balance = get_balance(self.keys, market_type=self.market_type)
        if account_balance < amount:
            return (
                f"Your balance of {self.currency} ({account_balance}) is not "
                f"large enough to buy {amount} {self.currency} worth of tokens."
            )

        # Exchange wxdai back to xdai if the balance is getting low, so we can keep paying for fees.
        if self.market_type == MarketType.OMEN:
            send_keeping_token_to_eoa_xdai(
                APIKeys(), OMEN_MIN_FEE_BALANCE, multiplier=MULTIPLIER
            )

        market: AgentMarket = self.market_type.market_class.get_binary_market(market_id)
        before_balance = market.get_token_balance(
            user_id=self.user_address,
            outcome=self.outcome,
        )

        market.buy_tokens(
            outcome=self.outcome,
            amount=amount,
        )
        after_balance = market.get_token_balance(
            user_id=self.user_address,
            outcome=self.outcome,
        )
        tokens = float(after_balance - before_balance)
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
    def __init__(self, market_type: MarketType, outcome: OutcomeStr, keys: APIKeys):
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

    def __call__(self, market_id: str, amount_outcome_tokens: float) -> str:
        amount = OutcomeToken(amount_outcome_tokens)

        # Exchange wxdai back to xdai if the balance is getting low, so we can keep paying for fees.
        if self.market_type == MarketType.OMEN:
            send_keeping_token_to_eoa_xdai(
                APIKeys(), OMEN_MIN_FEE_BALANCE, multiplier=MULTIPLIER
            )

        market: AgentMarket = self.market_type.market_class.get_binary_market(market_id)
        before_balance = market.get_token_balance(
            user_id=self.user_address,
            outcome=self.outcome,
        )

        market.sell_tokens(
            outcome=self.outcome,
            amount=amount,
        )

        after_balance = market.get_token_balance(
            user_id=self.user_address,
            outcome=self.outcome,
        )
        tokens = before_balance - after_balance
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
    @property
    def description(self) -> str:
        return f"Use this function to fetch your balance, given in USD"

    @property
    def example_args(self) -> list[str]:
        return []

    def __call__(self) -> float:
        return get_balance(self.keys, market_type=self.market_type).value


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
            larger_than=OutcomeToken(1e-4),  # Ignore very small positions
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


class GetKellyBet(MarketFunction):
    @property
    def description(self) -> str:
        return (
            "Use the Kelly Criterion to calculate the optimal bet size and "
            "direction for a binary market. Pass in the market_id and your "
            "estimated p_yes."
        )

    @property
    def example_args(self) -> list[float]:
        return [0.6, 0.5]

    def __call__(
        self,
        market_id: str,
        estimated_p_yes: float,
    ) -> str:
        confidence = 0.5  # Until confidence score is available, be conservative
        agent_market = self.market_type.market_class.get_binary_market(id=market_id)
        max_bet = agent_market.get_in_token(
            get_balance(self.keys, market_type=self.market_type)
        )
        kelly_bet = get_kelly_bet_simplified(
            market_p_yes=agent_market.p_yes,
            estimated_p_yes=estimated_p_yes,
            max_bet=max_bet,
            confidence=confidence,
        )
        kelly_bet_usd = agent_market.get_in_usd(kelly_bet.size)
        return (
            f"Bet size: {kelly_bet_usd.value:.2f} USD, "
            f"Bet direction: {kelly_bet.direction}"
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
    GetKellyBet,
]
