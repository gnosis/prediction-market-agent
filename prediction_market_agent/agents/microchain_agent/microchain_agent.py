from functions import MARKET_FUNCTIONS, MISC_FUNCTIONS
from microchain import LLM, Agent, Engine, OpenAIChatGenerator
from microchain.functions import Reasoning, Stop
from prediction_market_agent_tooling.markets.markets import MarketType

from prediction_market_agent.utils import APIKeys

engine = Engine()
engine.register(Reasoning())
engine.register(Stop())
for function in MISC_FUNCTIONS:
    engine.register(function())
for function in MARKET_FUNCTIONS:
    engine.register(function(market_type=MarketType.OMEN))

generator = OpenAIChatGenerator(
    model="gpt-4-turbo-preview",
    api_key=APIKeys().openai_api_key.get_secret_value(),
    api_base="https://api.openai.com/v1",
    temperature=0.7,
)
agent = Agent(llm=LLM(generator=generator), engine=engine)
agent.prompt = f"""Act as a agent. You can use the following functions:
 
{engine.help}
 
 
Only output valid Python function calls.
 
"""

agent.bootstrap = ['Reasoning("I need to reason step-by-step")']
agent.run(iterations=3)
generator.print_usage()
