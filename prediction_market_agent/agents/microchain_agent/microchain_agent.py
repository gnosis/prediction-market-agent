import os
import pprint

from dotenv import load_dotenv
from microchain import LLM, OpenAIChatGenerator
from prediction_market_agent_tooling.markets.agent_market import (
    AgentMarket,
    FilterBy,
    SortBy,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket

load_dotenv()

generator = OpenAIChatGenerator(
    model="gpt-4-turbo-preview",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base="https://api.openai.com/v1",
    temperature=0.7,
)

llm = LLM(generator=generator)

from microchain import Function

balance = 50
outcomeTokens = {}
outcomeTokens["Will Joe Biden get reelected in 2024?"] = {"yes": 0, "no": 0}
outcomeTokens["Will Bitcoin hit 100k in 2024?"] = {"yes": 0, "no": 0}


class Sum(Function):
    @property
    def description(self):
        return "Use this function to compute the sum of two numbers"

    @property
    def example_args(self):
        return [2, 2]

    def __call__(self, a: float, b: float):
        return a + b


class Product(Function):
    @property
    def description(self):
        return "Use this function to compute the product of two numbers"

    @property
    def example_args(self):
        return [2, 2]

    def __call__(self, a: float, b: float):
        return a * b


class GetMarkets(Function):
    @property
    def description(self):
        return "Use this function to get a list of predction markets and the current yes prices"

    @property
    def example_args(self):
        return []

    def __call__(self):
        # Get the 5 markets that are closing soonest
        markets: list[AgentMarket] = OmenAgentMarket.get_binary_markets(
            filter_by=FilterBy.OPEN,
            sort_by=SortBy.CLOSING_SOONEST,
            limit=5,
        )

        market_questions_and_prices = []
        for market in markets:
            market_questions_and_prices.append(market.question)
            market_questions_and_prices.append(market.p_yes)
        return market_questions_and_prices


class GetPropabilityForQuestion(Function):
    @property
    def description(self):
        return "Use this function to research the probability of an event occuring"

    @property
    def example_args(self):
        return ["Will Joe Biden get reelected in 2024?"]

    def __call__(self, a: str):
        if a == "Will Joe Biden get reelected in 2024?":
            return 0.41
        if a == "Will Bitcoin hit 100k in 2024?":
            return 0.22

        return 0.0


class GetBalance(Function):
    @property
    def description(self):
        return "Use this function to get your own balance in $"

    @property
    def example_args(self):
        return []

    def __call__(self):
        print(f"Your balance is: {balance} and ")
        pprint.pprint(outcomeTokens)
        return balance


class BuyYes(Function):
    @property
    def description(self):
        return "Use this function to buy yes outcome tokens of a prediction market. The second parameter specifies how much $ you spend."

    @property
    def example_args(self):
        return ["Will Joe Biden get reelected in 2024?", 2]

    def __call__(self, market: str, amount: int):
        global balance
        if amount > balance:
            return (
                f"Your balance of {balance} $ is not large enough to spend {amount} $."
            )

        balance -= amount
        return "Bought " + str(amount * 2) + " yes outcome token of: " + market


class BuyNo(Function):
    @property
    def description(self):
        return "Use this function to buy no outcome tokens of a prdiction market. The second parameter specifies how much $ you spend."

    @property
    def example_args(self):
        return ["Will Joe Biden get reelected in 2024?", 4]

    def __call__(self, market: str, amount: int):
        global balance
        if amount > balance:
            return (
                f"Your balance of {balance} $ is not large enough to spend {amount} $."
            )

        balance -= amount
        return "Bought " + str(amount * 2) + " no outcome token of: " + market


class SellYes(Function):
    @property
    def description(self):
        return "Use this function to sell yes outcome tokens of a prediction market. The second parameter specifies how much tokens you sell."

    @property
    def example_args(self):
        return ["Will Joe Biden get reelected in 2024?", 2]

    def __call__(self, market: str, amount: int):
        global outcomeTokens
        if amount > outcomeTokens[market]["yes"]:
            return f"Your balance of {outcomeTokens[market]['yes']} yes outcome tokens is not large enough to sell {amount}."

        outcomeTokens[market]["yes"] -= amount
        return "Sold " + str(amount) + " yes outcome token of: " + market


class SellNo(Function):
    @property
    def description(self):
        return "Use this function to sell no outcome tokens of a prdiction market. The second parameter specifies how much tokens you sell."

    @property
    def example_args(self):
        return ["Will Joe Biden get reelected in 2024?", 4]

    def __call__(self, market: str, amount: int):
        global outcomeTokens
        if amount > outcomeTokens[market]["no"]:
            return f"Your balance of {outcomeTokens[market]['no']} no outcome tokens is not large enough to sell {amount}."

        outcomeTokens[market]["no"] -= amount
        return "Sold " + str(amount) + " no outcome token of: " + market


class BalanceToOutcomes(Function):
    @property
    def description(self):
        return "Use this function to convert your balance into equal units of 'yes' and 'no' outcome tokens. The function takes the amount of balance as the argument."

    @property
    def example_args(self):
        return ["Will Joe Biden get reelected in 2024?", 50]

    def __call__(self, market: str, amount: int):
        global balance
        global outcomeTokens
        outcomeTokens[market]["yes"] += amount
        outcomeTokens[market]["no"] += amount
        balance -= amount
        return f"Converted {amount} units of balance into {amount} 'yes' outcome tokens and {amount} 'no' outcome tokens. Remaining balance is {balance}."


class SummarizeLearning(Function):
    @property
    def description(self):
        return "Use this function summarize your learnings and save them so that you can access them later."

    @property
    def example_args(self):
        return [
            "Today I learned that I need to check my balance fore making decisions about how much to invest."
        ]

    def __call__(self, summary: str):
        # print(summary)
        # pprint.pprint(outcomeTokens)
        return summary


from microchain import Agent, Engine
from microchain.functions import Reasoning, Stop

engine = Engine()
engine.register(Reasoning())
engine.register(Stop())
engine.register(Sum())
engine.register(Product())
engine.register(GetMarkets())
engine.register(GetPropabilityForQuestion())
engine.register(BuyNo())
engine.register(BuyYes())
# engine.register(SellNo())
# engine.register(SellYes())
engine.register(GetBalance())
# engine.register(BalanceToOutcomes())
engine.register(SummarizeLearning())


agent = Agent(llm=llm, engine=engine)
agent.prompt = f"""Act as a agent. You can use the following functions:
 
{engine.help}
 
 
Only output valid Python function calls.
 
"""

agent.bootstrap = ['Reasoning("I need to reason step-by-step")']
agent.run(iterations=10)
